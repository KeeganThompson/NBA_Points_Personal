import os
import glob
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import leaguegamelog

def evaluate_predictions(predictions_folder, game_date_str, season='2025-26'):
    """
    Evaluates CSV predictions against actual NBA box scores for a specific date.
    game_date_str format: 'MM/DD/YYYY' (e.g., '03/19/2026')
    """
    print(f"Fetching actual box scores from NBA API for {game_date_str}...")
    try:
        log = leaguegamelog.LeagueGameLog(
            season=season,
            date_from_nullable=game_date_str,
            date_to_nullable=game_date_str,
            player_or_team_abbreviation='P'
        ).get_data_frames()[0]
        
        log['PLAYER_NAME'] = log['PLAYER_NAME'].str.lower()
    except Exception as e:
        print(f"Failed to fetch API data: {e}")
        return

    csv_pattern = os.path.join(predictions_folder, "*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"No CSV files found in {predictions_folder}")
        return

    all_predictions = []
    
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            all_predictions.append(df)
        except Exception as e:
            print(f"Could not read {file}: {e}")
        
    combined_preds = pd.concat(all_predictions, ignore_index=True)
    combined_preds['Player_Lower'] = combined_preds['Player'].str.lower()
    
    merged = pd.merge(
        combined_preds, 
        log[['PLAYER_NAME', 'PTS']], 
        left_on='Player_Lower', 
        right_on='PLAYER_NAME', 
        how='inner'
    ).rename(columns={'PTS': 'Actual_PTS'})
    
    if merged.empty:
        print("No players matched the actual box scores. Check your date formatting.")
        return

    merged['Model_Error'] = (merged['Predicted_PTS'] - merged['Actual_PTS']).abs()
    merged['Baseline_Error'] = (merged['10_Game_Avg'] - merged['Actual_PTS']).abs()
    
    merged['Model_Direction'] = np.where(merged['Predicted_PTS'] > merged['10_Game_Avg'], 'Over', 'Under')
    merged['Actual_Direction'] = np.where(merged['Actual_PTS'] > merged['10_Game_Avg'], 'Over', 'Under')
    merged['Direction_Correct'] = (merged['Model_Direction'] == merged['Actual_Direction'])

    total_players = len(merged)
    model_mae = merged['Model_Error'].mean()
    baseline_mae = merged['Baseline_Error'].mean()
    within_3_pts = (merged['Model_Error'] <= 3.0).mean() * 100
    within_5_pts = (merged['Model_Error'] <= 5.0).mean() * 100
    directional_accuracy = merged['Direction_Correct'].mean() * 100
    
    print("\n" + "="*55)
    print(f"   PREDICTION EVALUATION REPORT: {game_date_str}   ")
    print("="*55)
    print(f"Total Players Evaluated (Played Minutes): {total_players}")
    print(f"-> Model Mean Absolute Error (MAE):     {model_mae:.2f} PTS")
    print(f"-> Baseline (10-Game Avg) MAE:          {baseline_mae:.2f} PTS")
    
    print("-" * 55)
    if model_mae < baseline_mae:
        improvement = baseline_mae - model_mae
        print(f" ✅ SUCCESS: Model BEAT the 10-game baseline by {improvement:.2f} points/player!")
    else:
        loss = model_mae - baseline_mae
        print(f" ❌ CAUTION: Model lost to the 10-game baseline by {loss:.2f} points/player.")
        
    print("-" * 55)
    print(f"Hit Rates:")
    print(f"-> Within +/- 3 Points:                 {within_3_pts:.1f}%")
    print(f"-> Within +/- 5 Points:                 {within_5_pts:.1f}%")
    print(f"-> Directional Accuracy (Over/Under):   {directional_accuracy:.1f}%")
    
    print("\nBiggest Model Misses (Top 5 - Use to debug):")
    worst_misses = merged.sort_values(by='Model_Error', ascending=False).head(5)
    for _, row in worst_misses.iterrows():
        print(f" - {row['Player']}: Predicted {row['Predicted_PTS']}, Actual {row['Actual_PTS']} (Off by {row['Model_Error']:.1f})")

if __name__ == '__main__':
    target_folder = r"Testing_Predictions/3_20_2026"
    target_date = "03/20/2026" 
    
    evaluate_predictions(target_folder, target_date)
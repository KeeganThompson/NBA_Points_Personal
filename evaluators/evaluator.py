import os
import sys
import glob
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import leaguegamelog

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

def evaluate_predictions(predictions_folder, game_date_str, season='2025-26'):
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
            if 'name' in df.columns and 'prediction' in df.columns:
                df = df.rename(columns={'name': 'Player', 'prediction': 'Predicted_PTS', 'avg_10': '10_Game_Avg'})
                
            if 'Player' in df.columns and 'Predicted_PTS' in df.columns:
                df['Player'] = df['Player'].str.lower()
                all_predictions.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            
    if not all_predictions:
        print("No valid prediction data found to evaluate.")
        return
        
    master_preds = pd.concat(all_predictions, ignore_index=True)
    
    merged = pd.merge(master_preds, log[['PLAYER_NAME', 'PTS']], left_on='Player', right_on='PLAYER_NAME', how='inner')
    merged = merged.rename(columns={'PTS': 'Actual_PTS'})
    
    if merged.empty:
        print("No players matched the box score. (Check your date format or if players actually played).")
        return
        
    merged['Error'] = merged['Predicted_PTS'] - merged['Actual_PTS']
    merged['Abs_Error'] = merged['Error'].abs()
    
    merged['Baseline_Error'] = merged['10_Game_Avg'] - merged['Actual_PTS']
    merged['Baseline_Abs_Error'] = merged['Baseline_Error'].abs()
    
    model_mae = merged['Abs_Error'].mean()
    baseline_mae = merged['Baseline_Abs_Error'].mean()
    
    total_players = len(merged)
    within_3_pts = (merged['Abs_Error'] <= 3).mean() * 100
    within_5_pts = (merged['Abs_Error'] <= 5).mean() * 100
    
    merged['Model_Direction'] = np.where(merged['Predicted_PTS'] > merged['10_Game_Avg'], 1, -1)
    merged['Actual_Direction'] = np.where(merged['Actual_PTS'] > merged['10_Game_Avg'], 1, -1)
    directional_accuracy = (merged['Model_Direction'] == merged['Actual_Direction']).mean() * 100
    
    print("\n" + "="*55)
    print(f"   PREDICTION EVALUATION REPORT: {game_date_str}   ")
    print("="*55)
    print(f"Total Players Evaluated (Played Minutes): {total_players}")
    print(f"-> Model Mean Absolute Error (MAE):     {model_mae:.2f} PTS")
    print(f"-> Baseline (10-Game Avg) MAE:          {baseline_mae:.2f} PTS")
    
    print("-" * 55)
    if model_mae < baseline_mae:
        improvement = baseline_mae - model_mae
        print(f"SUCCESS: Model BEAT the 10-game baseline by {improvement:.2f} points/player!")
    else:
        loss = model_mae - baseline_mae
        print(f"CAUTION: Model lost to the 10-game baseline by {loss:.2f} points/player.")
        
    print("-" * 55)
    print(f"Hit Rates:")
    print(f"-> Within +/- 3 Points:                 {within_3_pts:.1f}%")
    print(f"-> Within +/- 5 Points:                 {within_5_pts:.1f}%")
    print(f"-> Directional Accuracy (Over/Under):   {directional_accuracy:.1f}%")
    
    print("\nBiggest Model Misses (Top 5 - Use to debug):")
    worst_misses = merged.sort_values(by='Abs_Error', ascending=False).head(5)
    print(worst_misses[['Player', 'Predicted_PTS', 'Actual_PTS', '10_Game_Avg', 'Abs_Error']].to_string(index=False))

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python evaluator.py <path_to_predictions_folder> <date: MM/DD/YYYY>")
        print("Example: python evaluator.py Testing_Predictions/3-19-2026 03/19/2026")
    else:
        folder = sys.argv[1]
        date_str = sys.argv[2]
        evaluate_predictions(folder, date_str)
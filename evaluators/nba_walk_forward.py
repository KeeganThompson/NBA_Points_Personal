import os
import sys
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

# Suppress API and Pandas warnings for a clean console
warnings.filterwarnings('ignore')

from nba_api.stats.endpoints import leaguegamelog, leaguedashteamstats, commonplayerinfo

# Route to root directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from core.predictor import Predictor
from core.scraper import BasketballReferenceScraper

def fetch_season_cache():
    print(" Downloading 2025-26 Season Master Log (This takes ~5 seconds)...")
    season_log = leaguegamelog.LeagueGameLog(
        season='2025-26', player_or_team_abbreviation='P'
    ).get_data_frames()[0]
    
    season_log['GAME_DATE'] = pd.to_datetime(season_log['GAME_DATE'])
    
    # Build a quick team mapping
    team_map = dict(zip(season_log['TEAM_ABBREVIATION'], season_log['TEAM_ID']))
    
    print(" Fetching Advanced Team Stats...")
    try:
        adv_stats = leaguedashteamstats.LeagueDashTeamStats(
            measure_type_detailed_defense='Advanced', season='2025-26'
        ).get_data_frames()[0]
    except:
        adv_stats = pd.DataFrame()
        
    return season_log, team_map, adv_stats

def format_player_history(history_df):
    """Converts the NBA API log into the format your Predictor expects."""
    df = history_df.copy().sort_values('GAME_DATE')
    df['PTS'] = df['PTS']
    df['MP'] = df['MIN']
    df['FGA'] = df['FGA']
    df['Home'] = df['MATCHUP'].apply(lambda x: 1 if ' vs. ' in x else 0)
    df['Opp'] = df['MATCHUP'].apply(lambda x: x.split(' ')[-1])
    return df

def run_walk_forward():
    print("=======================================================")
    print("  INITIATING NBA HYBRID WALK-FORWARD BACKTESTER")
    print("=======================================================\n")
    
    vegas_files = []
    for directory in [BASE_DIR, os.path.join(BASE_DIR, 'Sportsbooks_Lines')]:
        if os.path.exists(directory):
            for file in os.listdir(directory):
                if file.startswith('vegas_props_') and file.endswith('.json'):
                    vegas_files.append(os.path.join(directory, file))
    
    if not vegas_files:
        print("[ERROR] No historical vegas_props_YYYY-MM-DD.json files found.")
        return
        
    vegas_files.sort()
    
    season_log, team_map, adv_stats = fetch_season_cache()
    predictor = Predictor()
    player_info_cache = {}
    
    total_bets = 0
    total_wins = 0
    total_losses = 0
    total_pushes = 0
    all_preds, all_actuals = [], []
    
    # Star Tier Trackers
    tier_stats = {
        5: {"W": 0, "L": 0, "P": 0},
        4: {"W": 0, "L": 0, "P": 0},
        3: {"W": 0, "L": 0, "P": 0}
    }

    # Time Machine Loop
    for file_path in vegas_files:
        date_str = os.path.basename(file_path).replace('vegas_props_', '').replace('.json', '')
        target_date = pd.to_datetime(date_str)
        
        print(f"\n=======================================================")
        print(f" DATE: {date_str}")
        print("=======================================================")
        
        with open(file_path, 'r') as f:
            vegas_data = json.load(f).get('lines', {})
            
        if not vegas_data: continue
            
        todays_games = season_log[season_log['GAME_DATE'] == target_date]
        if todays_games.empty:
            print("  [SKIP] No NBA games played on this date.")
            continue
            
        daily_bets = 0
        daily_wins = 0
        
        for index, row in todays_games.iterrows():
            player_name = row['PLAYER_NAME']
            player_id = row['PLAYER_ID']
            actual_pts = row['PTS']
            
            if player_name not in vegas_data: continue
                
            vegas_line = float(vegas_data[player_name])
            
            # Shielding future data
            history = season_log[(season_log['PLAYER_ID'] == player_id) & (season_log['GAME_DATE'] < target_date)]
            if len(history) < 5: continue 
                
            player_df = format_player_history(history)
            current_team_id = history.iloc[-1]['TEAM_ID']
            
            opp_abbr = row['MATCHUP'].split(' ')[-1]
            opp_id = team_map.get(opp_abbr, 0)
            is_home = 1 if ' vs. ' in row['MATCHUP'] else 0
            
            seven_days_ago = target_date - timedelta(days=7)
            games_in_7 = len(history[history['GAME_DATE'] >= seven_days_ago])
            
            next_game_data = {
                'Opp': opp_abbr, 'Opp_ID': opp_id, 'Home': is_home,
                'Date': date_str, 'Games_In_7_Days': games_in_7, 'Opp_Games_In_7_Days': 2.0
            }
            
            if player_id not in player_info_cache:
                try:
                    time.sleep(0.3)
                    info = commonplayerinfo.CommonPlayerInfo(player_id=player_id).get_data_frames()[0]
                    exp = int(info['SEASON_EXP'].iloc[0]) if not info.empty else 3
                    pos = info['POSITION'].iloc[0][0] if not info.empty and info['POSITION'].iloc[0] else 'F'
                    player_info_cache[player_id] = {'exp': exp, 'pos': pos}
                except:
                    player_info_cache[player_id] = {'exp': 3, 'pos': 'F'}
                    
            exp = player_info_cache[player_id]['exp']
            pos = player_info_cache[player_id]['pos']
            
            last_min_str = history.iloc[-1]['MIN']
            try: last_min = float(str(last_min_str).split(':')[0])
            except: last_min = 20.0
            is_starter = True if last_min > 24.0 else False

            try:
                result = predictor.predict_next_game(
                    player_df=player_df, adv_stats=adv_stats, team_map=team_map,
                    current_team_id=current_team_id, next_game_data=next_game_data,
                    experience=exp, position=pos, dvp_ranks={}, is_starter=is_starter
                )
                ai_pred = result['prediction']
            except Exception as e:
                continue
                
            # 5. Calculate Percentage Edge & Assign Star Rating
            diff = round(ai_pred - vegas_line, 1)
            pct_edge = abs(diff) / vegas_line if vegas_line > 0 else 0
            
            stars = 0
            if vegas_line < 12.0:
                # THE MICRO-LINE FIX: Require a minimum 2.0 point absolute difference for 5-Stars
                if pct_edge >= 0.25 and abs(diff) >= 2.0: stars = 5
                elif pct_edge >= 0.20: stars = 4
                elif pct_edge >= 0.16: stars = 3
            elif vegas_line < 22.0:
                # Added the 2.0 point safety floor here as well
                if pct_edge >= 0.18 and abs(diff) >= 2.0: stars = 5
                elif pct_edge >= 0.15: stars = 4
                elif pct_edge >= 0.12: stars = 3
            else: # Stars (22.0+)
                if abs(diff) >= 4.0: stars = 5
                elif abs(diff) >= 3.2: stars = 4
                elif abs(diff) >= 2.6: stars = 3

            if stars >= 3:
                # THE HYBRID LOGIC: Tail 5-Stars, Fade 3/4-Stars
                ai_raw_lean = "OVER" if diff > 0 else "UNDER"
                
                if stars == 5:
                    final_pick = ai_raw_lean
                    action = "TAIL"
                else:
                    final_pick = "UNDER" if ai_raw_lean == "OVER" else "OVER"
                    action = "FADE"
                
                daily_bets += 1
                total_bets += 1
                all_preds.append(ai_pred)
                all_actuals.append(actual_pts)
                
                # Grade the Final Pick
                if final_pick == "OVER":
                    res = "WIN" if actual_pts > vegas_line else "LOSS" if actual_pts < vegas_line else "PUSH"
                else:
                    res = "WIN" if actual_pts < vegas_line else "LOSS" if actual_pts > vegas_line else "PUSH"
                    
                if res == "WIN":
                    total_wins += 1; daily_wins += 1
                    tier_stats[stars]["W"] += 1
                elif res == "LOSS":
                    total_losses += 1
                    tier_stats[stars]["L"] += 1
                else:
                    total_pushes += 1
                    tier_stats[stars]["P"] += 1
                    
                marker = "✅" if res == "WIN" else "❌" if res == "LOSS" else "➖"
                star_str = "★" * stars + "☆" * (5 - stars)
                edge_display = f"+{diff}" if diff > 0 else f"{diff}"
                
                print(f"  {marker} [{star_str}] {player_name[:16]:<16} | Line: {vegas_line:<4} | AI: {ai_pred:<4.1f} | Pick: {final_pick:<5} ({action}) | Actual: {actual_pts:<2} -> {res}")

        if daily_bets > 0:
            print(f"\n   Daily Recap: {daily_wins}W - {daily_bets - daily_wins}L ({(daily_wins/daily_bets)*100:.1f}%)")

    # 6. Final Evaluation Report
    print("\n=======================================================")
    print("  FINAL HYBRID WALK-FORWARD RESULTS")
    print("=======================================================")
    if total_bets > 0:
        win_rate = total_wins / (total_wins + total_losses)
        mae = mean_absolute_error(all_actuals, all_preds)
        rmse = np.sqrt(mean_squared_error(all_actuals, all_preds))
        units = total_wins - (total_losses * 1.1)
        
        print(f"Total Premium Bets:    {total_bets}")
        print(f"Overall Record:        {total_wins}W - {total_losses}L - {total_pushes}P")
        print(f"True System Hit Rate:  {win_rate*100:.1f}%")
        print(f"Estimated Profit:      {units:+.2f} Units")
        print("-" * 55)
        
        # Breakdown by Star Rating
        for st in [5, 4, 3]:
            w = tier_stats[st]['W']
            l = tier_stats[st]['L']
            p = tier_stats[st]['P']
            t = w + l
            if t > 0:
                rate = (w / t * 100)
                st_units = w - (l * 1.1)
                strat = "(TAIL)" if st == 5 else "(FADE)"
                print(f"{st}-Star Bets {strat}: {w:>3}W - {l:<3}L ({rate:>4.1f}%) | {st_units:>+5.2f} U")
            else:
                print(f"{st}-Star Bets:        0W - 0L   (0.0%)  | +0.00 U")
                
        print("-" * 55)
        print(f"Model Mean Abs Error:  {mae:.2f} pts")
        print(f"Model RMSE:            {rmse:.2f} pts")
    else:
        print("No bets matched criteria during the testing period.")

if __name__ == "__main__":
    run_walk_forward()
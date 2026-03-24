import pandas as pd
import json
import os
import argparse
import unicodedata
from datetime import datetime
from scraper import BasketballReferenceScraper
from predictor import Predictor
from bet_analyzer import BetAnalyzer

def normalize_name(name):
    name = ''.join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')
    name = name.lower().replace('.', '').replace('-', ' ').replace("'", "")
    return name.replace(' jr', '').replace(' sr', '').replace(' iii', '').replace(' ii', '').strip()

def run_mega_slate(target_teams=None):
    print("=======================================================")
    print(" INITIALIZING NBA MICRO-SLATE GENERATOR")
    print("=======================================================\n")
    
    scraper = BasketballReferenceScraper()
    proc = Predictor()
    analyzer = BetAnalyzer()
    
    today_str = datetime.now().strftime('%Y-%m-%d')
    
    vegas_lines = {}
    if os.path.exists('vegas_props.json'):
        try:
            with open('vegas_props.json', 'r') as f:
                cache = json.load(f)
                vegas_lines = cache.get('lines', {})
        except: pass

    nba_teams = scraper.get_all_teams()
    team_map = {t['abbreviation']: t['id'] for t in nba_teams}
    team_map['BRK'] = team_map.get('BKN')
    team_map['CHO'] = team_map.get('CHA')
    team_map['PHO'] = team_map.get('PHX')
    
    teams_playing_today = []
    for t in nba_teams:
        abbr = t['abbreviation']
        next_game = scraper.scrape_next_game(abbr)
        if next_game and next_game['Date'] == today_str:
            teams_playing_today.append((abbr, next_game))
            
    # If predictor passed some teams, ignore the rest
    if target_teams:
        teams_playing_today = [t for t in teams_playing_today if t[0] in target_teams]
            
    if not teams_playing_today:
        print("No target games scheduled. Exiting.")
        return
        
    print(f"Fetching league stats for {len(teams_playing_today)} targeted teams...")
    
    adv_stats = scraper.scrape_advanced_team_stats()
    dvp_ranks = scraper.get_dvp_matrix()
    
    master_results = []
    
    for team_abbr, next_game in teams_playing_today:
        print(f"\nProcessing Team: {team_abbr} (vs {next_game['Opp']})")
        current_team_id = scraper.get_team_id(team_abbr)
        
        team_meta_map = scraper.get_player_metadata(team_abbr)
        projected_data = scraper.get_projected_lineup(team_abbr)
        
        if isinstance(projected_data, tuple) and len(projected_data) == 3:
            active_rotation, injured_out, projected_starters = projected_data
        else:
            active_rotation = projected_data[0] if isinstance(projected_data, tuple) else projected_data
            injured_out = []
            projected_starters = active_rotation[:5]
            
        if not active_rotation:
            continue
            
        all_player_data = scraper.get_bulk_player_gamelogs(active_rotation + injured_out)
        
        total_vacated_pts = 0.0
        for inj_player in injured_out:
            if inj_player in all_player_data and not all_player_data[inj_player].empty:
                season_avg = all_player_data[inj_player]['PTS'].mean()
                if pd.notna(season_avg):
                    total_vacated_pts += season_avg
                    
        final_player_list = [p for p in active_rotation if p in all_player_data]
        
        print(f"  Training AI Ensemble for {len(final_player_list)} players...")
        
        for player in final_player_list:
            try:
                player_data = all_player_data[player]
                metadata = team_meta_map.get(player, {'exp': 5, 'pos': 'F'})
                
                model_output = proc.predict_next_game(
                    player_data, adv_stats, team_map, current_team_id, next_game, 
                    metadata['exp'], metadata['pos'], dvp_ranks, player in projected_starters, 
                    vacated_pts=total_vacated_pts
                )
                
                recent_10 = player_data['PTS'].tail(10).mean()
                if pd.isna(recent_10): recent_10 = 0.0
                
                master_results.append({
                    "Player": player,
                    "Team": team_abbr,
                    "Opponent": next_game['Opp'],
                    "Predicted_PTS": round(model_output["prediction"], 1),
                    "Floor": round(model_output["floor"], 1),
                    "Ceiling": round(model_output["ceiling"], 1),
                    "10_Game_Avg": round(recent_10, 1)
                })
            except Exception: pass 

    # Add to the daily master CSV
    df = pd.DataFrame(master_results)
    
    archive_dir = "Mega_Slate_Predictions"
    os.makedirs(archive_dir, exist_ok=True)
    csv_filename = os.path.join(archive_dir, f"Master_Slate_{today_str}.csv")
    
    if os.path.exists(csv_filename):
        existing_df = pd.read_csv(csv_filename)
        # Drop players we just predicted so we can overwrite them with the fresh ones
        existing_df = existing_df[~existing_df['Player'].isin(df['Player'])]
        df = pd.concat([existing_df, df], ignore_index=True)
        
    df.to_csv(csv_filename, index=False)
    print(f"\n Micro-Slate appended. Database now holds {len(df)} total players.")
    
    # Auto-Scan for Bets for the players just ran
    print("\nScanning Micro-Slate for Vegas Edges...")
    new_bets = []
    for _, row in pd.DataFrame(master_results).iterrows():
        player = row['Player']
        pred = float(row['Predicted_PTS'])
        floor = float(row['Floor'])
        ceil = float(row['Ceiling'])
        
        v_line = vegas_lines.get(player)
        if v_line is None:
            clean_player = normalize_name(player)
            p_parts = clean_player.split()
            for v_name, line in vegas_lines.items():
                if normalize_name(v_name) == clean_player:
                    v_line = line
                    break
        
        if v_line is not None:
            edge = pred - v_line
            pick = "OVER" if edge > 0 else "UNDER"
            stars, reason = analyzer.calculate_confidence(pred, floor, ceil, v_line)
            
            if stars >= 3:
                color_edge = f"+{edge:.1f}" if edge > 0 else f"{edge:.1f}"
                print(f"{player:<22} | {v_line:<5.1f} | {pred:<7.1f} | {color_edge:<5} | {pick:<5} | {reason}")
                new_bets.append({
                    "Date": today_str,
                    "Player": player,
                    "Vegas_Line": v_line,
                    "AI_Pred": pred,
                    "AI_Floor": floor,
                    "AI_Ceiling": ceil,
                    "Pick": pick,
                    "Edge": round(edge, 1),
                    "Stars": stars,
                    "Actual_PTS": "PENDING",
                    "Result": "PENDING"
                })

    if new_bets:
        tracker_df = pd.DataFrame(new_bets)
        tracker_file = 'bet_tracker.csv'
        if os.path.exists(tracker_file):
            existing_df = pd.read_csv(tracker_file)
            existing_df = existing_df[~((existing_df['Date'] == today_str) & (existing_df['Player'].isin([b['Player'] for b in new_bets])))]
            tracker_df = pd.concat([existing_df, tracker_df], ignore_index=True)
        tracker_df.to_csv(tracker_file, index=False)
        print(f"📝 Logged {len(new_bets)} premium bets to tracker.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--teams', type=str, help='Comma separated list of team abbreviations')
    args = parser.parse_args()
    
    target_teams = args.teams.split(',') if args.teams else None
    run_mega_slate(target_teams)
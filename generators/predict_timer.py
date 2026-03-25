import time
import sys
import os
import subprocess
import json
from datetime import datetime, timedelta
from nba_api.stats.endpoints import scoreboardv2

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from core.scraper import BasketballReferenceScraper

def write_status(state, next_run=None, game_time=None, teams=None):
    status_file = os.path.join(BASE_DIR, 'timer_status.json')
    data = {
        "state": state,
        "next_run": next_run.isoformat() if next_run else None,
        "game_time": game_time.isoformat() if game_time else None,
        "teams": teams or []
    }
    with open(status_file, 'w') as f:
        json.dump(data, f)

def get_todays_schedule():
    print(f" Fetching live NBA schedule for {datetime.now().strftime('%Y-%m-%d')}...")
    try:
        board = scoreboardv2.ScoreboardV2().get_data_frames()[0]
    except Exception as e:
        print(f" Failed to reach NBA API: {e}")
        return {}
        
    scraper = BasketballReferenceScraper()
    nba_teams = scraper.get_all_teams()
    team_dict = {t['id']: t['abbreviation'] for t in nba_teams}
    
    schedule_map = {}
    for _, game in board.iterrows():
        home_id = game['HOME_TEAM_ID']
        away_id = game['VISITOR_TEAM_ID']
        home_abbr = team_dict.get(home_id)
        away_abbr = team_dict.get(away_id)
        status_text = str(game['GAME_STATUS_TEXT']).strip()
        
        if "Final" in status_text or "Half" in status_text or "Q" in status_text:
            continue 
            
        try:
            clean_time_str = status_text.replace(' ET', '').strip()
            parsed_time = datetime.strptime(clean_time_str, '%I:%M %p').time()
            now = datetime.now()
            game_datetime = datetime.combine(now.date(), parsed_time)
            
            if game_datetime not in schedule_map:
                schedule_map[game_datetime] = []
            if home_abbr not in schedule_map[game_datetime]:
                schedule_map[game_datetime].extend([home_abbr, away_abbr])
                print(f"   {away_abbr} @ {home_abbr} locked in for {parsed_time.strftime('%I:%M %p')}")
        except Exception as e:
            pass
    return schedule_map

def run_sniper():
    print("=======================================================")
    print("  INITIALIZING AUTOMATED SNIPER")
    print("=======================================================\n")
    
    schedule_map = get_todays_schedule()
    if not schedule_map:
        write_status("offline")
        print("No games found pending today.")
        return
        
    for game_time in sorted(schedule_map.keys()):
        teams = schedule_map[game_time]
        trigger_time = game_time - timedelta(minutes=30)
        
        while True:
            now = datetime.now()
            if now >= trigger_time and now < game_time:
                write_status("running", trigger_time, game_time, teams)
                print("\n=======================================================")
                print(f"  30-MINUTE WARNING: {game_time.strftime('%I:%M %p')} GAMES")
                print("=======================================================")
                
                target_teams_str = ",".join(teams)
                odds_script = os.path.join(BASE_DIR, 'generators', 'odds_fetcher.py')
                mega_script = os.path.join(BASE_DIR, 'generators', 'mega_slate.py')
                
                subprocess.run([sys.executable, odds_script, "--teams", target_teams_str])
                subprocess.run([sys.executable, mega_script, "--teams", target_teams_str])
                break
            elif now >= game_time:
                break
            else:
                write_status("sleeping", trigger_time, game_time, teams)
                time_to_trigger = (trigger_time - now).total_seconds()
                mins_left = int(time_to_trigger // 60)
                print(f"[{now.strftime('%I:%M %p')}] Sleeping. Next pull in {mins_left} mins...", end="\r")
                time.sleep(min(10, time_to_trigger)) 

    write_status("complete")
    print("\n=======================================================")
    print(" ALL GAMES PROCESSED.")
    print("=======================================================")

if __name__ == "__main__":
    run_sniper()
import time
import os
import subprocess
from datetime import datetime, timedelta
from nba_api.stats.endpoints import scoreboardv2
from scraper import BasketballReferenceScraper

def get_todays_schedule():
    print(f"Fetching live NBA schedule for {datetime.now().strftime('%Y-%m-%d')}...")
    
    try:
        board = scoreboardv2.ScoreboardV2().get_data_frames()[0]
    except Exception as e:
        print(f"Failed to reach NBA API: {e}")
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
                
            schedule_map[game_datetime].extend([home_abbr, away_abbr])
            print(f" {away_abbr} @ {home_abbr} locked in for {parsed_time.strftime('%I:%M %p')}")
            
        except Exception as e:
            print(f" Couldn't parse tip-off time for {away_abbr} @ {home_abbr}: '{status_text}'. Skipping prediction.")
            
    return schedule_map

def run_sniper():
    print("=======================================================")
    print(" INITIALIZING NBA PREDICTOR")
    print("=======================================================\n")
    
    schedule_map = get_todays_schedule()
    
    if not schedule_map:
        print("No games found pending today, or the schedule is empty.")
        return
        
    
    for game_time in sorted(schedule_map.keys()):
        teams = schedule_map[game_time]
        trigger_time = game_time - timedelta(minutes=30)
        
        while True:
            now = datetime.now()
            
            if now >= trigger_time and now < game_time:
                print("=======================================================")
                print(f" 30-MINUTE WARNING: {game_time.strftime('%I:%M %p')} GAMES")
                print(f" Teams: {', '.join(teams)}")
                print("=======================================================")
                
                # comma-separated string "MIA,LAL,BOS,NYK"
                target_teams_str = ",".join(teams)
                
                print("\n1️⃣ Fetching closing Vegas lines for targeted games...")
                subprocess.run(["python", "odds_fetcher.py", "--teams", target_teams_str])
                
                print("\n2️⃣ Launching Mega-Slate engine for targeted games...")
                subprocess.run(["python", "mega_slate.py", "--teams", target_teams_str])
                
                print(f"\nSequence complete for {game_time.strftime('%I:%M %p')} slate.\n")
                break
                
            elif now >= game_time:
                break
            
            else:
                time_to_trigger = (trigger_time - now).total_seconds()
                mins_left = int(time_to_trigger // 60)
                print(f"[{now.strftime('%I:%M %p')}] Sleeping. Next pull in {mins_left} minutes for {game_time.strftime('%I:%M %p')} games...", end="\r")
                time.sleep(min(30, time_to_trigger))

    print("\n=======================================================")
    print("ALL GAMES PROCESSED.")
    print("=======================================================")

if __name__ == "__main__":
    run_sniper()
import requests
import json
import os
import argparse
from datetime import datetime

API_KEY = '56280a5d9570359d7171919e38f88fbf'
SPORT = 'basketball_nba'
REGIONS = 'us'
MARKETS = 'player_points'

TEAM_MAPPING = {
    'ATL': 'Hawks', 'BOS': 'Celtics', 'BKN': 'Nets', 'CHI': 'Bulls', 'CHA': 'Hornets',
    'CLE': 'Cavaliers', 'DAL': 'Mavericks', 'DEN': 'Nuggets', 'DET': 'Pistons', 'GSW': 'Warriors',
    'HOU': 'Rockets', 'IND': 'Pacers', 'LAC': 'Clippers', 'LAL': 'Lakers', 'MEM': 'Grizzlies',
    'MIA': 'Heat', 'MIL': 'Bucks', 'MIN': 'Timberwolves', 'NOP': 'Pelicans', 'NYK': 'Knicks',
    'OKC': 'Thunder', 'ORL': 'Magic', 'PHI': '76ers', 'PHX': 'Suns', 'POR': 'Trail Blazers',
    'SAC': 'Kings', 'SAS': 'Spurs', 'TOR': 'Raptors', 'UTA': 'Jazz', 'WAS': 'Wizards',
    'BRK': 'Nets', 'CHO': 'Hornets', 'PHO': 'Suns'
}

def fetch_vegas_lines(target_teams=None):
    print("Connecting to The Odds API...")
    
    events_url = f"https://api.the-odds-api.com/v4/sports/{SPORT}/events"
    events_res = requests.get(events_url, params={'apiKey': API_KEY})
    
    if events_res.status_code != 200:
        print(f"Failed to fetch events: {events_res.text}")
        return

    events = events_res.json()
    
    target_mascots = []
    if target_teams:
        target_mascots = [TEAM_MAPPING.get(t, t).lower() for t in target_teams]
        print(f"Targeting specific games for: {', '.join(target_teams)}")
    else:
        print(f"Found {len(events)} upcoming events. Fetching all player props...")
        
    vegas_data = {}
    requests_used = 1

    for event in events:
        # If targeting specific teams, skip other games
        if target_mascots:
            event_name = f"{event.get('home_team', '')} {event.get('away_team', '')}".lower()
            if not any(mascot in event_name for mascot in target_mascots):
                continue
                
        event_id = event['id']
        
        odds_url = f"https://api.the-odds-api.com/v4/sports/{SPORT}/events/{event_id}/odds"
        odds_res = requests.get(odds_url, params={
            'apiKey': API_KEY,
            'regions': REGIONS,
            'markets': MARKETS
        })
        
        requests_used += 1
        
        if odds_res.status_code != 200:
            continue
            
        odds_json = odds_res.json()
        
        for bookmaker in odds_json.get('bookmakers', []):
            for market in bookmaker.get('markets', []):
                if market['key'] == 'player_points':
                    for outcome in market.get('outcomes', []):
                        player_name = outcome['description']
                        line = outcome.get('point')
                        
                        if line is not None:
                            if player_name not in vegas_data:
                                vegas_data[player_name] = []
                            vegas_data[player_name].append(line)

    final_lines = {}
    for player, lines in vegas_data.items():
        consensus_line = round(sum(lines) / len(lines), 1)
        final_lines[player] = consensus_line

    today_str = datetime.now().strftime('%Y-%m-%d')
    archive_dir = 'Sportsbooks_Lines'
    os.makedirs(archive_dir, exist_ok=True)
    archive_file = os.path.join(archive_dir, f'vegas_props_{today_str}.json')

    # Load prev fetched lines for today
    existing_lines = {}
    if os.path.exists(archive_file):
        try:
            with open(archive_file, 'r') as f:
                existing_lines = json.load(f).get('lines', {})
        except: pass
        
    # Append new lines
    existing_lines.update(final_lines)

    output_data = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "lines": existing_lines
    }

    with open('vegas_props.json', 'w') as f:
        json.dump(output_data, f, indent=4)
        
    with open(archive_file, 'w') as f:
        json.dump(output_data, f, indent=4)

    print(f"Successfully saved props for {len(final_lines)} targeted players!")
    print(f"Database now holds {len(existing_lines)} total lines for today.")
    print(f"API Requests Used This Run: {requests_used}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--teams', type=str, help='Comma separated list of team abbreviations (e.g. MIA,LAL)')
    args = parser.parse_args()
    
    teams_list = args.teams.split(',') if args.teams else None
    fetch_vegas_lines(teams_list)
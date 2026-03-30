import requests
import json
import os
import sys
import argparse
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

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
    if target_teams:
        print(f"Targeting specific games for: {', '.join(target_teams)}")

    events_res = requests.get(
        f"https://api.the-odds-api.com/v4/sports/{SPORT}/events",
        params={'apiKey': API_KEY}
    )
    
    if events_res.status_code != 200:
        print(f"Failed to fetch events: {events_res.text}")
        return

    events = events_res.json()
    vegas_data = {}
    api_calls = 1

    for event in events:
        home_team = event.get('home_team', '')
        away_team = event.get('away_team', '')
        
        home_abbr = next((abbr for abbr, name in TEAM_MAPPING.items() if name in home_team), None)
        away_abbr = next((abbr for abbr, name in TEAM_MAPPING.items() if name in away_team), None)

        if target_teams:
            if home_abbr not in target_teams and away_abbr not in target_teams:
                continue

        odds_res = requests.get(
            f"https://api.the-odds-api.com/v4/sports/{SPORT}/events/{event['id']}/odds",
            params={
                'apiKey': API_KEY,
                'regions': REGIONS,
                'markets': MARKETS
            }
        )
        api_calls += 1
        
        if odds_res.status_code != 200:
            continue
            
        odds_data = odds_res.json()
        
        for bookmaker in odds_data.get('bookmakers', []):
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
    archive_dir = os.path.join(BASE_DIR, 'Sportsbooks_Lines')
    os.makedirs(archive_dir, exist_ok=True)
    archive_file = os.path.join(archive_dir, f'vegas_props_{today_str}.json')

    existing_lines = {}
    if os.path.exists(archive_file):
        try:
            with open(archive_file, 'r') as f:
                existing_lines = json.load(f).get('lines', {})
        except: pass
        
    existing_lines.update(final_lines)

    output_data = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "lines": existing_lines
    }

    root_vegas_file = os.path.join(BASE_DIR, 'vegas_props.json')
    with open(root_vegas_file, 'w') as f:
        json.dump(output_data, f, indent=4)
        
    with open(archive_file, 'w') as f:
        json.dump(output_data, f, indent=4)

    print(f"Successfully saved props for {len(final_lines)} targeted players!")
    print(f"Database now holds {len(existing_lines)} total lines for today.")
    print(f"API Requests Used This Run: {api_calls}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Player Prop Lines from The Odds API")
    parser.add_argument('--teams', type=str, help='Comma separated list of team abbreviations to specifically target')
    args = parser.parse_args()
    
    target_teams = args.teams.split(',') if args.teams else None
    fetch_vegas_lines(target_teams)
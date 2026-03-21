import requests
import json
from datetime import datetime

# Your actual API Key
API_KEY = '56280a5d9570359d7171919e38f88fbf'
SPORT = 'basketball_nba'
REGIONS = 'us'
MARKETS = 'player_points'

def fetch_vegas_lines():
    print("🌐 Connecting to The Odds API...")
    
    # 1. Get all upcoming NBA events
    events_url = f"https://api.the-odds-api.com/v4/sports/{SPORT}/events"
    events_res = requests.get(events_url, params={'apiKey': API_KEY})
    
    if events_res.status_code != 200:
        print(f"❌ Failed to fetch events: {events_res.text}")
        return

    events = events_res.json()
    print(f"✅ Found {len(events)} upcoming events. Fetching player props...")
    
    vegas_data = {}
    requests_used = 1

    # 2. Get player props for each event
    for event in events:
        event_id = event['id']
        matchup = f"{event['away_team']} @ {event['home_team']}"
        
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
        
        # 3. Parse the sportsbooks to find the consensus line
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

    # 4. Average the lines and save
    final_lines = {}
    for player, lines in vegas_data.items():
        # Average the lines across all sportsbooks for the sharpest number
        consensus_line = round(sum(lines) / len(lines), 1)
        final_lines[player] = consensus_line

    # Save to a local cache file
    with open('vegas_props.json', 'w') as f:
        json.dump({
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "lines": final_lines
        }, f, indent=4)

    print(f"✅ Successfully saved props for {len(final_lines)} players to 'vegas_props.json'!")
    print(f"💸 API Requests Used This Run: {requests_used}")

if __name__ == "__main__":
    fetch_vegas_lines()
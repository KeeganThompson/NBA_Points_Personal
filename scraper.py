import pandas as pd
from datetime import datetime, timedelta
import time
import requests
from bs4 import BeautifulSoup

from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, teamgamelog, leaguedashteamstats, commonteamroster, leaguegamelog, boxscoretraditionalv2

try:
    from nba_api.stats.endpoints import scoreboardv3
    SCOREBOARD_CLASS = scoreboardv3.ScoreboardV3
    VERSION = 3
except ImportError:
    from nba_api.stats.endpoints import scoreboardv2
    SCOREBOARD_CLASS = scoreboardv2.ScoreboardV2
    VERSION = 2

class BasketballReferenceScraper:
    def __init__(self):
        self.team_mapping = {
            'ATL': 'Hawks', 'BOS': 'Celtics', 'BKN': 'Nets', 'CHI': 'Bulls', 'CHA': 'Hornets',
            'CLE': 'Cavaliers', 'DAL': 'Mavericks', 'DEN': 'Nuggets', 'DET': 'Pistons', 'GSW': 'Warriors',
            'HOU': 'Rockets', 'IND': 'Pacers', 'LAC': 'Clippers', 'LAL': 'Lakers', 'MEM': 'Grizzlies',
            'MIA': 'Heat', 'MIL': 'Bucks', 'MIN': 'Timberwolves', 'NOP': 'Pelicans', 'NYK': 'Knicks',
            'OKC': 'Thunder', 'ORL': 'Magic', 'PHI': '76ers', 'PHX': 'Suns', 'POR': 'Trail Blazers',
            'SAC': 'Kings', 'SAS': 'Spurs', 'TOR': 'Raptors', 'UTA': 'Jazz', 'WAS': 'Wizards'
        }

    def get_all_teams(self):
        return teams.get_teams()

    def get_team_id(self, abbr):
        nba_teams = teams.get_teams()
        for t in nba_teams:
            if t['abbreviation'] == abbr:
                return t['id']
        return None

    def get_team_experience_data(self, team_abbr, year=2026):
        """Pre-fetches experience levels for the entire team in one fast call."""
        team_id = self.get_team_id(team_abbr)
        season_str = f"{year-1}-{str(year)[-2:]}"
        try:
            time.sleep(0.5)
            roster = commonteamroster.CommonTeamRoster(team_id=team_id, season=season_str)
            df = roster.get_data_frames()[0]
            
            df.columns = [str(col).upper().replace('_', '') for col in df.columns]
            
            exp_map = {}
            player_col = 'PLAYER' if 'PLAYER' in df.columns else 'PLAYERNAME'
            exp_col = 'SEASONEXP' if 'SEASONEXP' in df.columns else 'EXP'
            
            if player_col in df.columns and exp_col in df.columns:
                for _, row in df.iterrows():
                    val = str(row[exp_col]).strip().upper()
                    exp_val = 0 if val == 'R' or val == '0' else int(float(val))
                    exp_map[row[player_col]] = exp_val
            return exp_map
        except Exception as e:
            print(f"Bulk experience fetch failed: {e}")
            return {}

    def get_active_roster(self, team_abbr, year=2026):
        team_id = self.get_team_id(team_abbr)
        if not team_id: return []
        season_str = f"{year-1}-{str(year)[-2:]}"
        
        try:
            time.sleep(1.0) 
            roster = commonteamroster.CommonTeamRoster(team_id=team_id, season=season_str)
            df = roster.get_data_frames()[0]
            if df.empty: return []
            return sorted(df['PLAYER'].tolist())
        except Exception as e:
            print(f"Active Roster fallback failed: {e}")
            return []

    def get_projected_lineup(self, team_abbr):
        mascot = self.team_mapping.get(team_abbr, "")
        url = "https://www.rotowire.com/basketball/nba-lineups.php"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        projected_starters = []
        if not mascot:
            return self.get_top_players_last_game(team_abbr, limit=5)

        active_roster = self.get_active_roster(team_abbr)
        
        try:
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                lineup_boxes = soup.find_all('div', class_='lineup')
                
                for box in lineup_boxes:
                    if mascot.lower() in box.text.lower():
                        player_elements = box.find_all('a', class_='lineup__player')
                        for p in player_elements:
                            clean_name = p.text.split('(')[0].replace(' Jr.', '').replace(' Sr.', '').replace(' III', '').replace(' II', '').strip()
                            
                            for roster_player in active_roster:
                                if clean_name.lower() in roster_player.lower():
                                    if roster_player not in projected_starters:
                                        projected_starters.append(roster_player)
                                    break
                                    
                        if projected_starters:
                            break
        except Exception as e:
            print(f"Web scraping failed: {e}")

        if len(projected_starters) < 5:
            fallback = self.get_top_players_last_game(team_abbr, limit=5)
            if not fallback:
                return active_roster 
            return fallback
            
        return projected_starters[:5]

    def get_top_players_last_game(self, team_abbr, year=2026, limit=5):
        team_id = self.get_team_id(team_abbr)
        season_str = f"{year-1}-{str(year)[-2:]}"
        
        try:
            time.sleep(1.0) 
            log = teamgamelog.TeamGameLog(team_id=team_id, season=season_str)
            df = log.get_data_frames()[0]
            if df.empty: return []
            
            game_id_col = 'Game_ID' if 'Game_ID' in df.columns else 'GAME_ID'
            last_game_id = df.iloc[0][game_id_col]
            
            time.sleep(1.0) 
            box = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=last_game_id)
            players_df = box.get_data_frames()[0]
            
            team_players = players_df[players_df['TEAM_ABBREVIATION'] == team_abbr].copy()
            if team_players.empty: 
                players_df['CLEAN_TEAM_ID'] = pd.to_numeric(players_df['TEAM_ID'], errors='coerce').fillna(0).astype(int)
                team_players = players_df[players_df['CLEAN_TEAM_ID'] == int(team_id)].copy()
                
            if team_players.empty: return []
                
            def parse_min(m):
                if pd.isna(m) or ':' not in str(m): return 0.0
                parts = str(m).split(':')
                return float(parts[0]) + (float(parts[1]) / 60.0)
                
            team_players['MIN_FLOAT'] = team_players['MIN'].apply(parse_min)
            top_players = team_players.sort_values(by='MIN_FLOAT', ascending=False).head(limit)
            
            return top_players['PLAYER_NAME'].tolist()
            
        except Exception:
            return []

    def get_bulk_player_gamelogs(self, roster_names, year=2026):
        season_curr = f"{year-1}-{str(year)[-2:]}"
        season_prev = f"{year-2}-{str(year-1)[-2:]}"
        
        time.sleep(1.0) 
        curr_log = leaguegamelog.LeagueGameLog(player_or_team_abbreviation='P', season=season_curr).get_data_frames()[0]
        
        time.sleep(0.5) 
        try:
            prev_log = leaguegamelog.LeagueGameLog(player_or_team_abbreviation='P', season=season_prev).get_data_frames()[0]
        except:
            prev_log = pd.DataFrame()
            
        all_logs = pd.concat([prev_log, curr_log], ignore_index=True)
        
        player_dict = {}
        for target_name in roster_names:
            
            match_df = all_logs[all_logs['PLAYER_NAME'].str.lower() == target_name.lower()]
            
            if match_df.empty:
                parts = target_name.lower().split()
                if len(parts) >= 2:
                    match_df = all_logs[
                        all_logs['PLAYER_NAME'].str.lower().str.contains(parts[0], regex=False) &
                        all_logs['PLAYER_NAME'].str.lower().str.contains(parts[-1], regex=False)
                    ]
            
            if not match_df.empty:
                api_exact_name = match_df.iloc[0]['PLAYER_NAME']
                api_df = all_logs[all_logs['PLAYER_NAME'] == api_exact_name].copy()
                
                api_df['GAME_DATE'] = pd.to_datetime(api_df['GAME_DATE'])
                api_df = api_df.sort_values('GAME_DATE').reset_index(drop=True)
                
                df_mapped = pd.DataFrame()
                df_mapped['GAME_DATE'] = api_df['GAME_DATE'] 
                df_mapped['G'] = range(1, len(api_df) + 1)
                df_mapped['Tm'] = api_df['MATCHUP'].apply(lambda x: x.split(' ')[0])
                df_mapped['Home'] = api_df['MATCHUP'].apply(lambda x: 0 if '@' in x else 1)
                df_mapped['Opp'] = api_df['MATCHUP'].apply(lambda x: x.split(' ')[-1])
                df_mapped['WL'] = api_df['WL']
                df_mapped['GS'] = 1 
                df_mapped['MP'] = api_df['MIN'].apply(lambda x: f"{x}:00" if ":" not in str(x) else str(x))
                df_mapped['FGA'] = api_df['FGA']
                df_mapped['FG%'] = api_df['FG_PCT']
                df_mapped['3PA'] = api_df['FG3A']
                df_mapped['3P%'] = api_df['FG3_PCT']
                df_mapped['FTA'] = api_df['FTA']
                df_mapped['FT%'] = api_df['FT_PCT']
                df_mapped['ORB'] = api_df['OREB']
                df_mapped['DRB'] = api_df['DREB']
                df_mapped['TRB'] = api_df['REB']
                df_mapped['AST'] = api_df['AST']
                df_mapped['STL'] = api_df['STL']
                df_mapped['BLK'] = api_df['BLK']
                df_mapped['TOV'] = api_df['TOV']
                df_mapped['PF'] = api_df['PF']
                df_mapped['PTS'] = api_df['PTS']
                
                player_dict[api_exact_name] = df_mapped
            
        return player_dict

    def scrape_advanced_team_stats(self, year=2026):
        season_str = f"{year-1}-{str(year)[-2:]}"
        
        time.sleep(1.0) 
        stats = leaguedashteamstats.LeagueDashTeamStats(season=season_str, measure_type_detailed_defense='Advanced')
        df = stats.get_data_frames()[0]
        
        cols = ['TEAM_ID', 'DEF_RATING']
        
        if 'PACE' in df.columns: cols.append('PACE')
        else: df['PACE'] = 100.0; cols.append('PACE')
            
        if 'NET_RATING' in df.columns: cols.append('NET_RATING')
        else: df['NET_RATING'] = 0.0; cols.append('NET_RATING')
            
        return df[cols]

    def scrape_next_game(self, team_abbr, year=2026):
        nba_teams = teams.get_teams()
        team_id = self.get_team_id(team_abbr)
        if not team_id: return None

        today = datetime.now()
        for i in range(14):
            date_str = (today + timedelta(days=i)).strftime('%Y-%m-%d')
            try:
                sb = SCOREBOARD_CLASS(game_date=date_str)
                data_frames = sb.get_data_frames()
            except Exception:
                continue 
            
            for df in data_frames:
                if df.empty: continue
                df.columns = [str(col).upper().replace('_', '') for col in df.columns]
                
                if 'HOMETEAMID' in df.columns:
                    away_col = 'VISITORTEAMID' if 'VISITORTEAMID' in df.columns else 'AWAYTEAMID'
                    for _, game in df.iterrows():
                        home_id = game.get('HOMETEAMID')
                        away_id = game.get(away_col)
                        
                        if team_id in [home_id, away_id]:
                            is_home = (team_id == home_id)
                            opp_id = away_id if is_home else home_id
                            opp_abbr = next((t['abbreviation'] for t in nba_teams if t['id'] == opp_id), "UNK")
                            return {"Opp": opp_abbr, "Opp_ID": opp_id, "Home": 1 if is_home else 0, "Date": date_str}
                            
                elif 'TEAMID' in df.columns and 'GAMEID' in df.columns:
                    if team_id in df['TEAMID'].values:
                        game_id = df[df['TEAMID'] == team_id]['GAMEID'].iloc[0]
                        game_teams = df[df['GAMEID'] == game_id]
                        opp_row = game_teams[game_teams['TEAMID'] != team_id]
                        
                        if not opp_row.empty:
                            opp_id = opp_row['TEAMID'].iloc[0]
                            opp_abbr = next((t['abbreviation'] for t in nba_teams if t['id'] == opp_id), "UNK")
                            return {"Opp": opp_abbr, "Opp_ID": opp_id, "Home": 1, "Date": date_str}
            time.sleep(0.3)
        return None
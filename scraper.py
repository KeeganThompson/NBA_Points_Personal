import pandas as pd
from datetime import datetime, timedelta
import time
import requests
from bs4 import BeautifulSoup
import unicodedata

from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, teamgamelog, leaguedashteamstats, commonteamroster, leaguegamelog, boxscoretraditionalv2

try:
    from nba_api.stats.endpoints import scoreboardv3
    SCOREBOARD_CLASS = scoreboardv3.ScoreboardV3
except ImportError:
    from nba_api.stats.endpoints import scoreboardv2
    SCOREBOARD_CLASS = scoreboardv2.ScoreboardV2

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
        for t in teams.get_teams():
            if t['abbreviation'] == abbr:
                return t['id']
        return None

    def get_team_experience_data(self, team_abbr, year=2026):
        team_id = self.get_team_id(team_abbr)
        season_str = f"{year-1}-{str(year)[-2:]}"
        try:
            time.sleep(0.5)
            df = commonteamroster.CommonTeamRoster(team_id=team_id, season=season_str).get_data_frames()[0]
            
            exp_map = {}
            cols = [str(col).upper().replace('_', '') for col in df.columns]
            df.columns = cols
            
            player_col = next((c for c in cols if c in ['PLAYER', 'PLAYERNAME']), None)
            exp_col = next((c for c in cols if c in ['SEASONEXP', 'EXP']), None)
            
            if player_col and exp_col:
                for _, row in df.iterrows():
                    val = str(row[exp_col]).strip().upper()
                    exp_val = 0 if val in ['R', '0'] else int(float(val))
                    exp_map[row[player_col]] = exp_val
            return exp_map
        except:
            return {}

    def get_active_roster(self, team_abbr, year=2026):
        team_id = self.get_team_id(team_abbr)
        if not team_id: return []
        season_str = f"{year-1}-{str(year)[-2:]}"
        
        try:
            time.sleep(1.0) 
            df = commonteamroster.CommonTeamRoster(team_id=team_id, season=season_str).get_data_frames()[0]
            if df.empty: return []
            player_col = next((c for c in df.columns if c.upper().replace('_', '') in ['PLAYER', 'PLAYERNAME']), 'PLAYER')
            return sorted(df[player_col].tolist())
        except Exception:
            return []

    def get_injured_players(self, team_abbr):
        """Scrapes ESPN and explicitly identifies players marked 'Out'."""
        mascot = self.team_mapping.get(team_abbr, "")
        url = "https://www.espn.com/nba/injuries"
        headers = {"User-Agent": "Mozilla/5.0"}
        out_players = []
        
        try:
            res = requests.get(url, headers=headers, timeout=10)
            if res.status_code == 200:
                soup = BeautifulSoup(res.content, 'html.parser')
                titles = soup.find_all(class_='Table__Title')
                target_table = None
                
                for title in titles:
                    if mascot.lower() in title.text.lower() or team_abbr.lower() in title.text.lower():
                        target_table = title.find_next('table')
                        break
                
                if target_table:
                    for row in target_table.find_all('tr'):
                        cols = row.find_all('td')
                        if len(cols) >= 4:
                            name = cols[0].text.strip()
                            status = cols[3].text.strip()
                            if 'Out' in status: 
                                out_players.append(name)
        except Exception as e:
            print(f"ESPN Scrape Error: {e}")
            
        return out_players

    def _fuzzy_match(self, name1, name2):
        """Safely matches ESPN names (e.g., 'L. Ball') with NBA API names ('LaMelo Ball')"""
        def clean(s):
            s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
            return s.lower().replace('.', '').replace('-', ' ').replace(' jr', '').replace(' sr', '').strip()
            
        c1, c2 = clean(name1), clean(name2)
        if c1 == c2: return True
        p1, p2 = c1.split(), c2.split()
        if len(p1) >= 2 and len(p2) >= 2:
            return p1[0][0] == p2[0][0] and p1[-1] == p2[-1]
        return False

    def get_projected_lineup(self, team_abbr):
        mascot = self.team_mapping.get(team_abbr, "")
        url = "https://www.rotowire.com/basketball/nba-lineups.php"
        headers = {"User-Agent": "Mozilla/5.0"}
        
        projected_starters = []
        active_roster = self.get_active_roster(team_abbr)
        espn_injuries = self.get_injured_players(team_abbr)
        
        injured_out = []
        for inj in espn_injuries:
            for rp in active_roster:
                if self._fuzzy_match(inj, rp) and rp not in injured_out:
                    injured_out.append(rp)
                    break

        has_lineup = False
        try:
            res = requests.get(url, headers=headers, timeout=5)
            if res.status_code == 200:
                soup = BeautifulSoup(res.content, 'html.parser')
                lineup_boxes = soup.find_all('div', class_='lineup')
                for box in lineup_boxes:
                    if mascot.lower() in box.text.lower():
                        player_tags = box.find_all('a', class_='lineup__player')
                        if len(player_tags) > 0:
                            has_lineup = True
                            for p in player_tags:
                                for rp in active_roster:
                                    if self._fuzzy_match(p.text, rp) and rp not in projected_starters:
                                        projected_starters.append(rp)
                                        break
                        break
        except Exception:
            pass

        final_rotation = []
        
        if has_lineup and len(projected_starters) > 0:
            for p in projected_starters:
                if p not in injured_out and p not in final_rotation:
                    final_rotation.append(p)
            
            if len(final_rotation) < 5:
                last_game = self.get_top_players_last_game(team_abbr, limit=10)
                for p in last_game:
                    if p not in injured_out and p not in final_rotation:
                        final_rotation.append(p)
                    if len(final_rotation) >= 5:
                        break
                        
        else:
            last_game = self.get_top_players_last_game(team_abbr, limit=15)
            for p in last_game:
                if p not in injured_out:
                    final_rotation.append(p)
                if len(final_rotation) >= 10:
                    break
                    
            if len(final_rotation) < 5:
                for p in active_roster:
                    if p not in injured_out and p not in final_rotation:
                        final_rotation.append(p)
                    if len(final_rotation) >= 10:
                        break
                        
        return final_rotation, injured_out

    def get_top_players_last_game(self, team_abbr, year=2026, limit=15):
        team_id = self.get_team_id(team_abbr)
        season_str = f"{year-1}-{str(year)[-2:]}"
        
        try:
            time.sleep(1.0) 
            df = teamgamelog.TeamGameLog(team_id=team_id, season=season_str).get_data_frames()[0]
            if df.empty: return []
            
            game_id_col = next((c for c in df.columns if c.upper().replace('_', '') == 'GAMEID'), 'Game_ID')
            last_game_id = df.iloc[0][game_id_col]
            
            time.sleep(1.0) 
            players_df = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=last_game_id).get_data_frames()[0]
            
            team_abbrev_col = next((c for c in players_df.columns if c.upper().replace('_', '') == 'TEAMABBREVIATION'), 'TEAM_ABBREVIATION')
            team_id_col = next((c for c in players_df.columns if c.upper().replace('_', '') == 'TEAMID'), 'TEAM_ID')
            
            team_players = players_df[players_df[team_abbrev_col] == team_abbr].copy()
            if team_players.empty: 
                players_df['CLEAN_TEAM_ID'] = pd.to_numeric(players_df[team_id_col], errors='coerce').fillna(0).astype(int)
                team_players = players_df[players_df['CLEAN_TEAM_ID'] == int(team_id)].copy()
                
            if team_players.empty: return []
                
            def parse_min(m):
                if pd.isna(m) or ':' not in str(m): return 0.0
                parts = str(m).split(':')
                return float(parts[0]) + (float(parts[1]) / 60.0)
                
            team_players['MIN_FLOAT'] = team_players['MIN'].apply(parse_min)
            top_players = team_players.sort_values(by='MIN_FLOAT', ascending=False).head(limit)
            
            player_name_col = next((c for c in top_players.columns if c.upper().replace('_', '') in ['PLAYERNAME', 'PLAYER']), 'PLAYER_NAME')
            return top_players[player_name_col].tolist()
            
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
        player_col = next((c for c in all_logs.columns if c.upper().replace('_', '') in ['PLAYERNAME', 'PLAYER']), 'PLAYER_NAME')
        
        player_dict = {}
        for target_name in roster_names:
            match_df = all_logs[all_logs[player_col].str.lower() == target_name.lower()]
            
            if match_df.empty:
                parts = target_name.lower().split()
                if len(parts) >= 2:
                    match_df = all_logs[
                        all_logs[player_col].str.lower().str.contains(parts[0], regex=False) &
                        all_logs[player_col].str.lower().str.contains(parts[-1], regex=False)
                    ]
            
            if not match_df.empty:
                api_exact_name = match_df.iloc[0][player_col]
                api_df = all_logs[all_logs[player_col] == api_exact_name].copy()
                
                date_col = next((c for c in api_df.columns if c.upper().replace('_', '') == 'GAMEDATE'), 'GAME_DATE')
                api_df['GAMEDATE_CLN'] = pd.to_datetime(api_df[date_col])
                api_df = api_df.sort_values('GAMEDATE_CLN').reset_index(drop=True)
                
                df_mapped = pd.DataFrame()
                df_mapped['GAME_DATE'] = api_df['GAMEDATE_CLN'] 
                df_mapped['Opp'] = api_df['MATCHUP'].apply(lambda x: x.split(' ')[-1])
                df_mapped['Home'] = api_df['MATCHUP'].apply(lambda x: 0 if '@' in x else 1)
                df_mapped['MP'] = api_df['MIN']
                df_mapped['FGA'] = api_df['FGA']
                df_mapped['FTA'] = api_df['FTA']
                df_mapped['TOV'] = api_df['TOV']
                df_mapped['PTS'] = api_df['PTS']
                
                player_dict[api_exact_name] = df_mapped
            
        return player_dict

    def scrape_advanced_team_stats(self, year=2026):
        season_str = f"{year-1}-{str(year)[-2:]}"
        time.sleep(1.0) 
        df = leaguedashteamstats.LeagueDashTeamStats(season=season_str, measure_type_detailed_defense='Advanced').get_data_frames()[0]
        
        res = pd.DataFrame()
        res['TEAM_ID'] = df['TEAM_ID'] if 'TEAM_ID' in df.columns else df['TEAMID'] if 'TEAMID' in df.columns else None
        res['DEF_RATING'] = df['DEF_RATING'] if 'DEF_RATING' in df.columns else df['DEFRATING'] if 'DEFRATING' in df.columns else 115.0
        res['PACE'] = df['PACE'] if 'PACE' in df.columns else 100.0
        res['NET_RATING'] = df['NET_RATING'] if 'NET_RATING' in df.columns else df['NETRATING'] if 'NETRATING' in df.columns else 0.0
        
        return res

    def scrape_next_game(self, team_abbr, year=2026):
        nba_teams = teams.get_teams()
        team_id = self.get_team_id(team_abbr)
        if not team_id: return None

        today = datetime.now()
        for i in range(14):
            date_str = (today + timedelta(days=i)).strftime('%Y-%m-%d')
            try:
                data_frames = SCOREBOARD_CLASS(game_date=date_str).get_data_frames()
            except Exception:
                continue 
            
            for df in data_frames:
                if df.empty: continue
                
                home_col = next((c for c in df.columns if c.upper().replace('_', '') == 'HOMETEAMID'), 'HOME_TEAM_ID')
                away_col = next((c for c in df.columns if c.upper().replace('_', '') in ['VISITORTEAMID', 'AWAYTEAMID']), 'VISITOR_TEAM_ID')
                team_col = next((c for c in df.columns if c.upper().replace('_', '') == 'TEAMID'), 'TEAM_ID')
                game_col = next((c for c in df.columns if c.upper().replace('_', '') == 'GAMEID'), 'GAME_ID')
                
                if home_col in df.columns and away_col in df.columns:
                    for _, game in df.iterrows():
                        home_id = game.get(home_col)
                        away_id = game.get(away_col)
                        
                        if team_id in [home_id, away_id]:
                            is_home = (team_id == home_id)
                            opp_id = away_id if is_home else home_id
                            opp_abbr = next((t['abbreviation'] for t in nba_teams if t['id'] == opp_id), "UNK")
                            return {"Opp": opp_abbr, "Opp_ID": opp_id, "Home": 1 if is_home else 0, "Date": date_str}
                            
                elif team_col in df.columns and game_col in df.columns:
                    if team_id in df[team_col].values:
                        game_id = df[df[team_col] == team_id][game_col].iloc[0]
                        game_teams = df[df[game_col] == game_id]
                        opp_row = game_teams[game_teams[team_col] != team_id]
                        
                        if not opp_row.empty:
                            opp_id = opp_row[team_col].iloc[0]
                            opp_abbr = next((t['abbreviation'] for t in nba_teams if t['id'] == opp_id), "UNK")
                            return {"Opp": opp_abbr, "Opp_ID": opp_id, "Home": 1, "Date": date_str}
            time.sleep(0.3)
        return None
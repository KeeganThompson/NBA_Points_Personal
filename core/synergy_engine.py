import pandas as pd
import warnings
from nba_api.stats.endpoints import leaguegamelog
from datetime import datetime

warnings.filterwarnings('ignore')

class SynergyEngine:
    def __init__(self):
        self.cached_log = None
        self.season_str = None
        self._set_season()

    def _set_season(self):
        now = datetime.now()
        year = now.year
        self.season_str = f"{year-1}-{str(year)[-2:]}" if now.month < 10 else f"{year}-{str(year+1)[-2:]}"

    def get_log(self):
        if self.cached_log is None:
            try:
                log = leaguegamelog.LeagueGameLog(player_or_team_abbreviation='P', season=self.season_str).get_data_frames()[0]
                log.columns = [str(c).upper().replace('_', '') for c in log.columns]
                self.cached_log = log
            except Exception as e:
                print(f"Synergy Engine API Error: {e}")
                return pd.DataFrame()
        return self.cached_log

    def get_scorer_correlations(self, team_abbr, scorer_name, scraper):
        log_df = self.get_log()
        if log_df.empty: return []
        
        team_log = log_df[log_df['TEAMABBREVIATION'] == team_abbr.upper()].copy()
        if team_log.empty: return []

        try:
            projected = scraper.get_projected_lineup(team_abbr)
            active_roster = projected[0] if isinstance(projected, tuple) else projected
        except:
            active_roster = team_log['PLAYERNAME'].unique().tolist()

        if not active_roster: return []

        team_log = team_log[team_log['PLAYERNAME'].isin(active_roster)]
        if team_log.empty: return []

        player_mins = team_log.groupby('PLAYERNAME')['MIN'].sum().sort_values(ascending=False)
        top_players = player_mins.head(10).index.tolist()
        
        if scorer_name not in top_players and scorer_name in team_log['PLAYERNAME'].values:
            top_players.append(scorer_name)

        team_log = team_log[team_log['PLAYERNAME'].isin(top_players)]
        
        pts_pivot = team_log.pivot(index='GAMEID', columns='PLAYERNAME', values='PTS').fillna(0)
        ast_pivot = team_log.pivot(index='GAMEID', columns='PLAYERNAME', values='AST').fillna(0)

        if scorer_name not in pts_pivot.columns: return []

        scorer_pts = pts_pivot[scorer_name]
        
        results = []
        for creator in top_players:
            if creator == scorer_name or creator not in ast_pivot.columns: continue
            creator_ast = ast_pivot[creator]
            
            corr = creator_ast.corr(scorer_pts)
            
            if pd.notna(corr) and (corr >= 0.200 or corr <= -0.200):
                if corr >= 0.4: strength = " Strong Positive"
                elif corr >= 0.2: strength = " Moderate Positive"
                elif corr <= -0.4: strength = " Strong Negative"
                else: strength = " Moderate Negative"
                
                results.append({
                    "teammate": creator,
                    "correlation": round(corr, 3),
                    "strength": strength,
                    "advice": "OVER" if corr > 0 else "UNDER"
                })
                
        results.sort(key=lambda x: abs(x['correlation']), reverse=True)
        return results
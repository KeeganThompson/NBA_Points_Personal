import pandas as pd
import argparse
import sys
import os
import warnings
from nba_api.stats.endpoints import leaguegamelog
from nba_api.stats.static import teams
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from core.scraper import BasketballReferenceScraper

warnings.filterwarnings('ignore')

def analyze_correlations(team_abbr, log_df, scraper):
    team_log = log_df[log_df['TEAMABBREVIATION'] == team_abbr.upper()].copy()
    if team_log.empty:
        print(f" No data found for team '{team_abbr}'.")
        return

    active_roster = scraper.get_active_roster(team_abbr)
    
    if not active_roster:
        print(f"Could not fetch active roster for {team_abbr}. Skipping.")
        return

    team_log = team_log[team_log['PLAYERNAME'].isin(active_roster)]
    
    if team_log.empty:
        print(f" No matching logs found for the active roster of {team_abbr}.")
        return

    player_mins = team_log.groupby('PLAYERNAME')['MIN'].sum().sort_values(ascending=False)
    top_players = player_mins.head(8).index.tolist()
    
    team_log = team_log[team_log['PLAYERNAME'].isin(top_players)]
    
    pts_pivot = team_log.pivot(index='GAMEID', columns='PLAYERNAME', values='PTS').fillna(0)
    ast_pivot = team_log.pivot(index='GAMEID', columns='PLAYERNAME', values='AST').fillna(0)
    
    print(f"\n=======================================================")
    print(f"  CURRENT TEAMMATE SYNERGY: {team_abbr.upper()} (Top 8 Active)")
    print(f"=======================================================")
    
    for creator in top_players:
        if creator not in ast_pivot.columns: continue
        creator_ast = ast_pivot[creator]
        
        correlations = []
        for scorer in top_players:
            if creator == scorer or scorer not in pts_pivot.columns: continue
            scorer_pts = pts_pivot[scorer]
            
            corr = creator_ast.corr(scorer_pts)
            
            if pd.notna(corr) and (corr >= 0.200 or corr <= -0.200):
                correlations.append((scorer, corr))
                
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        if correlations:
            print(f"\n When {creator} gets ASSISTS, how are teammates affected?")
            for scorer, corr in correlations:
                if corr >= 0.4:
                    strength = " Strong Positive"
                elif corr >= 0.2:
                    strength = " Moderate Positive"
                elif corr <= -0.4:
                    strength = " Strong Negative"
                else:
                    strength = " Moderate Negative"
                    
                print(f"   -> {scorer:<22}: {corr:+.3f} ({strength})")

def run_synergy(target_team):
    now = datetime.now()
    year = now.year
    season_str = f"{year-1}-{str(year)[-2:]}" if now.month < 10 else f"{year}-{str(year+1)[-2:]}"
    
    print(f" Fetching {season_str} season game logs from NBA API...")
    try:
        log = leaguegamelog.LeagueGameLog(player_or_team_abbreviation='P', season=season_str).get_data_frames()[0]
        log.columns = [str(c).upper().replace('_', '') for c in log.columns]
    except Exception as e:
        print(f" API Error: {e}")
        return

    scraper = BasketballReferenceScraper()

    if target_team.lower() == 'all':
        for t in teams.get_teams():
            analyze_correlations(t['abbreviation'], log, scraper)
    else:
        analyze_correlations(target_team, log, scraper)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Assist-to-Points correlations between teammates.")
    parser.add_argument('--team', type=str, required=True, help="Team Abbreviation (e.g., LAL) or 'ALL'")
    args = parser.parse_args()
    
    run_synergy(args.team)
import pandas as pd
import numpy as np
import time
import warnings
from datetime import datetime, timedelta
from nba_api.stats.endpoints import leaguegamelog, commonteamroster
from predictor import Predictor

warnings.filterwarnings('ignore')

class WalkForwardBacktester:
    def __init__(self, days_to_test=10, year=2026):
        self.days_to_test = days_to_test
        self.year = year
        self.season_str = f"{year-1}-{str(year)[-2:]}"
        self.proc = Predictor()
        self.master_log = None
        self.player_meta = {}
        self.team_map = {}
        
    def _simplify_pos(self, p):
        p = str(p).upper()
        if 'G' in p: return 'G'
        if 'C' in p: return 'C'
        return 'F'

    def fetch_master_data(self):
        print("📥 Downloading full season baseline (This takes ~10 seconds)...")
        time.sleep(1)
        self.master_log = leaguegamelog.LeagueGameLog(
            player_or_team_abbreviation='P', season=self.season_str
        ).get_data_frames()[0]
        self.master_log.columns = [str(col).upper().replace('_', '') for col in self.master_log.columns]
        self.master_log['GAMEDATE'] = pd.to_datetime(self.master_log['GAMEDATE'])
        
        team_log = leaguegamelog.LeagueGameLog(
            player_or_team_abbreviation='T', season=self.season_str
        ).get_data_frames()[0]
        team_log.columns = [str(col).upper().replace('_', '') for col in team_log.columns]
        for _, row in team_log.iterrows():
            abbr = row['TEAMABBREVIATION']
            tid = row['TEAMID']
            if abbr not in self.team_map:
                self.team_map[abbr] = tid

        print("📥 Fetching Player Positions and Metadata (Pulling 30 rosters - takes ~15s)...")
        for abbr, tid in self.team_map.items():
            try:
                time.sleep(0.4)
                roster = commonteamroster.CommonTeamRoster(team_id=tid, season=self.season_str).get_data_frames()[0]
                cols = [str(col).upper().replace('_', '') for col in roster.columns]
                roster.columns = cols
                
                player_col = next((c for c in cols if c in ['PLAYER', 'PLAYERNAME']), None)
                pos_col = next((c for c in cols if c == 'POSITION'), None)
                exp_col = next((c for c in cols if c in ['SEASONEXP', 'EXP']), None)
                
                if player_col and pos_col:
                    for _, row in roster.iterrows():
                        p_name = row[player_col]
                        p_pos = self._simplify_pos(row[pos_col])
                        
                        exp_val = 5
                        if exp_col:
                            val = str(row[exp_col]).strip().upper()
                            if val in ['R', '0']: 
                                exp_val = 0
                            else:
                                try: exp_val = int(float(val))
                                except: exp_val = 5
                                
                        self.player_meta[p_name] = {
                            'pos': p_pos,
                            'exp': exp_val
                        }
            except Exception as e:
                pass

    def calculate_point_in_time_dvp(self, historical_log):
        """Calculates 1-30 DvP using ONLY data that existed before the target date"""
        valid_logs = historical_log.dropna(subset=['PTS']).copy()
        valid_logs['OPP'] = valid_logs['MATCHUP'].apply(lambda x: str(x).split(' ')[-1])
        valid_logs['POS'] = valid_logs['PLAYERNAME'].map(lambda x: self.player_meta.get(x, {}).get('pos', 'F'))
        
        game_totals = valid_logs.groupby(['GAMEID', 'OPP', 'POS'])['PTS'].sum().reset_index()
        team_avg = game_totals.groupby(['OPP', 'POS'])['PTS'].mean().reset_index()
        
        dvp_ranks = {}
        for pos in ['G', 'F', 'C']:
            pos_data = team_avg[team_avg['POS'] == pos].copy()
            pos_data['RANK'] = pos_data['PTS'].rank(ascending=True, method='min')
            for _, row in pos_data.iterrows():
                opp = row['OPP']
                if opp not in dvp_ranks: dvp_ranks[opp] = {}
                dvp_ranks[opp][pos] = row['RANK']
        return dvp_ranks

    def calculate_point_in_time_adv_stats(self, historical_log):
        """Estimates Team Net Rating based on point differentials up to the target date"""
        team_log = historical_log.groupby(['TEAMID', 'GAMEID']).agg({'PTS': 'sum', 'MIN': 'sum'}).reset_index()
        
        res = pd.DataFrame()
        team_avg = team_log.groupby('TEAMID')['PTS'].mean().reset_index()
        res['TEAM_ID'] = team_avg['TEAMID']
        res['PACE'] = 100.0
        res['NET_RATING'] = (team_avg['PTS'] - 110.0)
        return res

    def run_backtest(self):
        if self.master_log is None:
            self.fetch_master_data()
            
        max_date = self.master_log['GAMEDATE'].max()
        start_date = max_date - timedelta(days=self.days_to_test)
        
        print(f"\n🚀 STARTING WALK-FORWARD BACKTEST")
        print(f"Testing from {start_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}\n")
        
        results = []
        
        for i in range(self.days_to_test + 1):
            target_date = start_date + timedelta(days=i)
            
            past_data = self.master_log[self.master_log['GAMEDATE'] < target_date].copy()
            todays_games = self.master_log[self.master_log['GAMEDATE'] == target_date].copy()
            
            if todays_games.empty:
                continue
                
            print(f"⏳ Simulating {target_date.strftime('%Y-%m-%d')} ({len(todays_games)} player logs found)...")
            
            dvp_ranks = self.calculate_point_in_time_dvp(past_data)
            adv_stats = self.calculate_point_in_time_adv_stats(past_data)
            
            for _, row in todays_games.iterrows():
                player_name = row['PLAYERNAME']
                actual_pts = row['PTS']
                current_team_id = row['TEAMID']
                matchup = row['MATCHUP']
                
                opp_abbr = matchup.split(' ')[-1]
                is_home = 1 if '@' not in matchup else 0
                opp_id = self.team_map.get(opp_abbr, None)
                
                player_hist = past_data[past_data['PLAYERNAME'] == player_name].copy()
                if len(player_hist) < 5:
                    continue 
                    
                player_hist = player_hist.sort_values('GAMEDATE').reset_index(drop=True)
                player_df = pd.DataFrame()
                player_df['GAME_DATE'] = player_hist['GAMEDATE']
                player_df['Opp'] = player_hist['MATCHUP'].apply(lambda x: x.split(' ')[-1])
                player_df['Home'] = player_hist['MATCHUP'].apply(lambda x: 0 if '@' in x else 1)
                player_df['MP'] = player_hist['MIN']
                player_df['FGA'] = player_hist['FGA']
                player_df['FTA'] = player_hist['FTA']
                player_df['TOV'] = player_hist['TOV']
                player_df['PTS'] = player_hist['PTS']
                
                next_game_data = {
                    "Opp": opp_abbr, "Opp_ID": opp_id, "Home": is_home, "Date": target_date.strftime('%Y-%m-%d')
                }
                
                meta = self.player_meta.get(player_name, {'pos': 'F', 'exp': 5})
                
                try:
                    preds = self.proc.predict_next_game(
                        player_df, adv_stats, self.team_map, current_team_id, 
                        next_game_data, meta['exp'], meta['pos'], dvp_ranks, is_starter=False
                    )
                    
                    recent_10 = player_df['PTS'].tail(10).mean()
                    
                    results.append({
                        'Date': target_date.strftime('%Y-%m-%d'),
                        'Player': player_name,
                        'Predicted': preds['prediction'],
                        'Floor': preds['floor'],
                        'Ceiling': preds['ceiling'],
                        '10_Game_Avg': recent_10,
                        'Actual': actual_pts
                    })
                except Exception as e:
                    pass
                    
        df_res = pd.DataFrame(results)
        df_res['Error'] = abs(df_res['Predicted'] - df_res['Actual'])
        df_res['Base_Error'] = abs(df_res['10_Game_Avg'] - df_res['Actual'])
        
        df_res['Bet_Signal'] = 'NO BET'
        df_res.loc[(df_res['Predicted'] - df_res['10_Game_Avg']) >= 2.5, 'Bet_Signal'] = 'OVER'
        df_res.loc[(df_res['10_Game_Avg'] - df_res['Predicted']) >= 2.5, 'Bet_Signal'] = 'UNDER'
        
        bets_df = df_res[df_res['Bet_Signal'] != 'NO BET'].copy()
        bets_df['Won'] = False
        bets_df.loc[(bets_df['Bet_Signal'] == 'OVER') & (bets_df['Actual'] > bets_df['10_Game_Avg']), 'Won'] = True
        bets_df.loc[(bets_df['Bet_Signal'] == 'UNDER') & (bets_df['Actual'] < bets_df['10_Game_Avg']), 'Won'] = True
        
        total_bets = len(bets_df)
        wins = bets_df['Won'].sum()
        win_rate = (wins / total_bets) * 100 if total_bets > 0 else 0.0
        
        profit = (wins * 90.90) - ((total_bets - wins) * 100)
        
        print("\n=======================================================")
        print(" 📈 BACKTESTING RESULTS (LAST 10 DAYS)")
        print("=======================================================")
        print(f"Total Player Games Simulated: {len(df_res)}")
        print(f"AI Ensemble MAE:              {df_res['Error'].mean():.2f} PTS")
        print(f"Baseline (10-Game Avg) MAE:   {df_res['Base_Error'].mean():.2f} PTS")
        print("-------------------------------------------------------")
        print(" 💰 HYPOTHETICAL BETTING SIMULATION (-110 Odds)")
        print(" Strategy: Bet $100 when AI prediction differs from 10-Game Avg by 2.5+ pts")
        print("-------------------------------------------------------")
        print(f"Total Bets Placed:  {total_bets}")
        print(f"Total Wins:         {wins}")
        print(f"Win Rate:           {win_rate:.1f}%  (Break-even is 52.38%)")
        print(f"Net Profit:         ${profit:.2f}")
        print("=======================================================\n")
        
        df_res.to_csv("Backtest_Results.csv", index=False)
        print("Full breakdown saved to 'Backtest_Results.csv'.")

if __name__ == "__main__":
    bt = WalkForwardBacktester(days_to_test=10)
    bt.run_backtest()
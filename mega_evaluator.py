import pandas as pd
import numpy as np
import argparse
import os
import glob
import json
import unicodedata
from datetime import datetime
from nba_api.stats.endpoints import leaguegamelog
import warnings

warnings.filterwarnings('ignore')

class MegaEvaluator:
    def __init__(self):
        self.archive_dir = 'Sportsbooks_Lines'
        self.slate_dir = 'Mega_Slate_Predictions'

    def _normalize(self, name):
        name = ''.join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')
        name = name.lower().replace('.', '').replace('-', ' ').replace("'", "")
        return name.replace(' jr', '').replace(' sr', '').replace(' iii', '').replace(' ii', '').strip()

    def convert_minutes(self, x):
        if pd.isna(x): return 0.0
        if isinstance(x, (int, float)): return float(x)
        x_str = str(x).strip()
        try:
            if ":" in x_str:
                return float(x_str.split(":")[0]) + (float(x_str.split(":")[1]) / 60.0)
            return float(x_str)
        except: return 0.0

    def get_season_string(self, target_date):
        year = target_date.year
        if target_date.month >= 10:
            return f"{year}-{str(year+1)[-2:]}"
        else:
            return f"{year-1}-{str(year)[-2:]}"

    def fetch_actuals(self, season_str, api_date_str=None):
        try:
            log = leaguegamelog.LeagueGameLog(
                player_or_team_abbreviation='P',
                season=season_str,
                date_from_nullable=api_date_str,
                date_to_nullable=api_date_str
            ).get_data_frames()[0]
            log.columns = [str(c).upper().replace('_', '') for c in log.columns]
            return log
        except Exception as e:
            print(f" API Error: Failed to fetch box scores: {e}")
            return pd.DataFrame()

    def process_single_date(self, csv_filepath, log_df):
        try:
            df = pd.read_csv(csv_filepath)
            filename = os.path.basename(csv_filepath)
            date_str = filename.replace('Master_Slate_', '').replace('.csv', '')
            target_date = datetime.strptime(date_str, '%Y-%m-%d')
        except Exception as e:
            return pd.DataFrame()

        vegas_file = os.path.join(self.archive_dir, f'vegas_props_{date_str}.json')
        if not os.path.exists(vegas_file):
            if os.path.exists('vegas_props.json') and date_str == datetime.now().strftime('%Y-%m-%d'):
                vegas_file = 'vegas_props.json'
            else:
                return pd.DataFrame()

        with open(vegas_file, 'r') as f:
            vegas_data = json.load(f).get('lines', {})

        df['Vegas_Line'] = np.nan
        for idx, row in df.iterrows():
            player = row['Player']
            v_line = vegas_data.get(player)
            if v_line is None:
                clean_player = self._normalize(player)
                for v_name, line in vegas_data.items():
                    if self._normalize(v_name) == clean_player:
                        v_line = line
                        break
            df.at[idx, 'Vegas_Line'] = v_line

        df = df.dropna(subset=['Vegas_Line'])
        
        # Filter log for this specific date
        api_date_str = target_date.strftime('%Y-%m-%d')
        daily_log = log_df[log_df['GAMEDATE'] == api_date_str].copy() if 'GAMEDATE' in log_df.columns else log_df.copy()

        daily_log['PLAYER_LOWER'] = daily_log['PLAYERNAME'].str.lower()
        df['PLAYER_LOWER'] = df['Player'].str.lower()
        
        merged = pd.merge(df, daily_log[['PLAYER_LOWER', 'PTS', 'MIN']], on='PLAYER_LOWER', how='inner')
        merged = merged.rename(columns={'PTS': 'Actual_PTS'})
        
        merged['MIN_FLOAT'] = merged['MIN'].apply(self.convert_minutes)
        merged = merged[merged['MIN_FLOAT'] >= 5.0].copy()
        merged['Date'] = date_str
        
        return merged

    def print_dashboard(self, merged, title):
        if merged.empty:
            print(f" No valid actionable data found for {title}.")
            return

        merged['AI_Error'] = abs(merged['Predicted_PTS'] - merged['Actual_PTS'])
        merged['Vegas_Error'] = abs(merged['Vegas_Line'] - merged['Actual_PTS'])
        
        ai_mae = merged['AI_Error'].mean()
        vegas_mae = merged['Vegas_Error'].mean()

        merged['AI_Edge'] = merged['Predicted_PTS'] - merged['Vegas_Line']
        merged['Bet_Signal'] = 'NO BET'
        
        merged.loc[merged['AI_Edge'] >= 2.5, 'Bet_Signal'] = 'OVER'
        merged.loc[merged['AI_Edge'] <= -2.5, 'Bet_Signal'] = 'UNDER'
        
        bets = merged[merged['Bet_Signal'] != 'NO BET'].copy()
        bets['Result'] = 'LOSS'
        
        bets.loc[(bets['Bet_Signal'] == 'OVER') & (bets['Actual_PTS'] > bets['Vegas_Line']), 'Result'] = 'WIN'
        bets.loc[(bets['Bet_Signal'] == 'UNDER') & (bets['Actual_PTS'] < bets['Vegas_Line']), 'Result'] = 'WIN'
        bets.loc[bets['Actual_PTS'] == bets['Vegas_Line'], 'Result'] = 'PUSH'
        
        total_bets = len(bets)
        wins = len(bets[bets['Result'] == 'WIN'])
        pushes = len(bets[bets['Result'] == 'PUSH'])
        losses = len(bets[bets['Result'] == 'LOSS'])
        
        actionable_bets = wins + losses
        win_rate = (wins / actionable_bets * 100) if actionable_bets > 0 else 0.0
        profit = (wins * 90.90) - (losses * 100)

        print("\n=======================================================")
        print(f"  MEGA-SLATE PERFORMANCE REPORT: {title}")
        print("=======================================================")
        print(f"Total Players Evaluated Against DraftKings: {len(merged)}")
        print("-------------------------------------------------------")
        print(f"AI Ensemble MAE:              {ai_mae:.2f} PTS")
        print(f"Vegas (DraftKings) MAE:       {vegas_mae:.2f} PTS")
        
        if ai_mae < vegas_mae:
            print(f" AI beat Vegas mathematically by {(vegas_mae - ai_mae):.2f} points per player!")
        else:
            print(f" Vegas beat the AI by {(ai_mae - vegas_mae):.2f} points.")
            
        print("-------------------------------------------------------")
        print("  HYPOTHETICAL BETTING SIMULATION (-110 Odds)")
        print(" Strategy: Bet WITH the AI when it disagrees with Vegas by 2.5+ pts")
        print("-------------------------------------------------------")
        print(f"Total Bets Placed:  {total_bets} ({pushes} Pushes)")
        print(f"Record:             {wins}W - {losses}L")
        print(f"Win Rate:           {win_rate:.1f}%  (Break-even is 52.38%)")
        print(f"Net Profit:         ${profit:.2f}")
        print("=======================================================\n")

    def evaluate_file(self, csv_filepath):
        date_str = os.path.basename(csv_filepath).replace('Master_Slate_', '').replace('.csv', '')
        target_date = datetime.strptime(date_str, '%Y-%m-%d')
        season_str = self.get_season_string(target_date)
    
        print(f" Fetching actual NBA box scores for {date_str}...")
        log_df = self.fetch_actuals(season_str, target_date.strftime('%m/%d/%Y'))
        
        merged = self.process_single_date(csv_filepath, log_df)
        self.print_dashboard(merged, date_str)

    def evaluate_all(self):
        print(f"=======================================================")
        print(f"  INITIALIZING MASS HISTORICAL EVALUATION")
        print(f"=======================================================")
        
        csv_files = glob.glob(os.path.join(self.slate_dir, 'Master_Slate_*.csv'))
        if not csv_files:
            print("No Master Slates found in Mega_Slate_Predictions folder.")
            return
            
        print(f" Found {len(csv_files)} historical slates. Fetching ENTIRE SEASON box scores...")
        season_str = self.get_season_string(datetime.now())
        full_log = self.fetch_actuals(season_str)
        
        if full_log.empty:
            return

        all_merged_dfs = []
        for file in csv_files:
            date_str = os.path.basename(file).replace('Master_Slate_', '').replace('.csv', '')
            merged = self.process_single_date(file, full_log)
            if not merged.empty:
                all_merged_dfs.append(merged)
                print(f"  Processed {date_str} ({len(merged)} players)")
            else:
                print(f"  Skipped {date_str} (Missing Vegas Lines or Box Scores)")

        if not all_merged_dfs:
            print("Could not map any historical slates to Vegas lines.")
            return
            
        master_merged = pd.concat(all_merged_dfs, ignore_index=True)
        self.print_dashboard(master_merged, "ALL HISTORICAL DATA (AGGREGATE)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a Mega-Slate CSV against actual NBA box scores and Vegas lines.")
    parser.add_argument('--file', type=str, help="Path to a specific Master_Slate CSV file.")
    parser.add_argument('--gradeall', action='store_true', help="Grade ALL historical slates in the directory.")
    args = parser.parse_args()
    
    evaluator = MegaEvaluator()
    if args.gradeall:
        evaluator.evaluate_all()
    elif args.file:
        evaluator.evaluate_file(args.file)
    else:
        print("Error: You must specify either --file <path> or --gradeall")
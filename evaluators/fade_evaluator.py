import pandas as pd
import numpy as np
import argparse
import os
import sys
import glob
import json
import unicodedata
from datetime import datetime
from nba_api.stats.endpoints import leaguegamelog
import warnings

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

warnings.filterwarnings('ignore')

class FadeEvaluator:
    def __init__(self):
        self.archive_dir = os.path.join(BASE_DIR, 'Sportsbooks_Lines')
        self.slate_dir = os.path.join(BASE_DIR, 'Mega_Slate_Predictions')

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
        except:
            return pd.DataFrame()

    def process_single_date(self, csv_filepath, log_df):
        try:
            df = pd.read_csv(csv_filepath)
            date_str = os.path.basename(csv_filepath).replace('Master_Slate_', '').replace('.csv', '')
            target_date = datetime.strptime(date_str, '%Y-%m-%d')
        except:
            return pd.DataFrame()

        if 'Predicted_PTS' in df.columns:
            df = df.rename(columns={'Predicted_PTS': 'Pred_PTS'})

        vegas_file = os.path.join(self.archive_dir, f'vegas_props_{date_str}.json')
        if not os.path.exists(vegas_file):
            root_vegas = os.path.join(BASE_DIR, 'vegas_props.json')
            if os.path.exists(root_vegas) and date_str == datetime.now().strftime('%Y-%m-%d'):
                vegas_file = root_vegas
            else:
                return pd.DataFrame()

        with open(vegas_file, 'r') as f:
            vegas_data = json.load(f).get('lines', {})

        df['Vegas_Line'] = np.nan
        for idx, row in df.iterrows():
            player = row['Player']
            v_data = vegas_data.get(player, {})
            if not v_data:
                clean_player = self._normalize(player)
                for v_name, data in vegas_data.items():
                    if self._normalize(v_name) == clean_player:
                        v_data = data
                        break
                        
            if isinstance(v_data, (float, int)):
                df.at[idx, 'Vegas_Line'] = float(v_data)
            elif isinstance(v_data, dict):
                df.at[idx, 'Vegas_Line'] = v_data.get('PTS')

        api_date_str = target_date.strftime('%Y-%m-%d')
        daily_log = log_df[log_df['GAMEDATE'] == api_date_str].copy() if 'GAMEDATE' in log_df.columns else log_df.copy()
        
        if daily_log.empty:
            return pd.DataFrame()

        daily_log['PLAYER_LOWER'] = daily_log['PLAYERNAME'].str.lower()
        df['PLAYER_LOWER'] = df['Player'].str.lower()
        
        merged = pd.merge(df, daily_log[['PLAYER_LOWER', 'PTS', 'MIN']], on='PLAYER_LOWER', how='inner')
        merged = merged.rename(columns={'PTS': 'Actual_PTS'})
        merged['MIN_FLOAT'] = merged['MIN'].apply(self.convert_minutes)
        merged = merged[merged['MIN_FLOAT'] >= 5.0].copy()
        
        return merged

    def print_dashboard(self, merged, title):
        if merged.empty:
            return

        m_df = merged.dropna(subset=['Vegas_Line', 'Pred_PTS']).copy()
        if m_df.empty: return
            
        m_df['AI_Edge'] = m_df['Pred_PTS'] - m_df['Vegas_Line']
        m_df['Bet_Signal'] = 'NO BET'
        
        # Fade
        m_df.loc[m_df['AI_Edge'] >= 2.5, 'Bet_Signal'] = 'UNDER (Fade AI Over)'
        m_df.loc[m_df['AI_Edge'] <= -2.5, 'Bet_Signal'] = 'OVER (Fade AI Under)'
        
        bets = m_df[m_df['Bet_Signal'] != 'NO BET'].copy()
        bets['Result'] = 'LOSS'
        bets.loc[(bets['Bet_Signal'] == 'UNDER (Fade AI Over)') & (bets['Actual_PTS'] < bets['Vegas_Line']), 'Result'] = 'WIN'
        bets.loc[(bets['Bet_Signal'] == 'OVER (Fade AI Under)') & (bets['Actual_PTS'] > bets['Vegas_Line']), 'Result'] = 'WIN'
        bets.loc[bets['Actual_PTS'] == bets['Vegas_Line'], 'Result'] = 'PUSH'
        
        wins = len(bets[bets['Result']=='WIN'])
        pushes = len(bets[bets['Result']=='PUSH'])
        losses = len(bets[bets['Result']=='LOSS'])
        actionable = wins + losses
        win_rate = (wins / actionable * 100) if actionable > 0 else 0.0
        profit = (wins * 90.90) - (losses * 100)

        print("\n=======================================================")
        print(f"  FADE STRATEGY REPORT: {title}")
        print("=======================================================")
        print(f"Total Bets Placed:  {len(bets)} ({pushes} Pushes)")
        print(f"Overall Record:     {wins}W - {losses}L")
        print(f"Win Rate:           {win_rate:.1f}%  (Break-even is 52.38%)")
        print(f"Net Profit:         ${profit:.2f}")
        print("=======================================================\n")

    def evaluate_file(self, csv_filepath):
        date_str = os.path.basename(csv_filepath).replace('Master_Slate_', '').replace('.csv', '')
        target_date = datetime.strptime(date_str, '%Y-%m-%d')
        season_str = self.get_season_string(target_date)
        
        log_df = self.fetch_actuals(season_str, target_date.strftime('%m/%d/%Y'))
        merged = self.process_single_date(csv_filepath, log_df)
        self.print_dashboard(merged, date_str)

    def evaluate_all(self):
        print(f"=======================================================")
        print(f"  INITIALIZING MASS HISTORICAL FADE EVALUATION")
        print(f"=======================================================")
        
        csv_files = glob.glob(os.path.join(self.slate_dir, 'Master_Slate_*.csv'))
        if not csv_files: return
            
        season_str = self.get_season_string(datetime.now())
        full_log = self.fetch_actuals(season_str)
        if full_log.empty: return

        all_merged = []
        for file in csv_files:
            merged = self.process_single_date(file, full_log)
            if not merged.empty: all_merged.append(merged)

        if all_merged: self.print_dashboard(pd.concat(all_merged, ignore_index=True), "ALL HISTORICAL DATA")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, help="Specific Master_Slate CSV file.")
    parser.add_argument('--gradeall', action='store_true')
    args = parser.parse_args()
    
    if args.gradeall: FadeEvaluator().evaluate_all()
    elif args.file: FadeEvaluator().evaluate_file(args.file)
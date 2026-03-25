import pandas as pd
import numpy as np
import argparse
import os
import glob
import json
import unicodedata
from datetime import datetime, timedelta
from nba_api.stats.endpoints import leaguegamelog
import warnings

warnings.filterwarnings('ignore')

class BetAnalyzer:
    def __init__(self):
        self.tracker_file = 'bet_tracker.csv'
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

    def calculate_confidence(self, pred, floor, ceiling, v_line):
        """Calculates the Star Rating of a bet based on mathematical boundaries."""
        edge = pred - v_line
        
        if edge >= 2.5:
            if floor > v_line:
                return 5, "Floor > Vegas Line (Absolute Lock)"
            elif edge >= 4.0:
                return 4, "Massive Median Edge (+4.0)"
            else:
                return 3, "Standard Median Edge (+2.5)"
        elif edge <= -2.5:
            if ceiling < v_line:
                return 5, "Ceiling < Vegas Line (Absolute Lock)"
            elif edge <= -4.0:
                return 4, "Massive Median Edge (-4.0)"
            else:
                return 3, "Standard Median Edge (-2.5)"
                
        return 0, "No Edge"

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

    def grade_tracker(self):
        """Standard daily function to grade PENDING bets in bet_tracker.csv"""
        print("=======================================================")
        print("  DAILY BET TRACKER GRADING")
        print("=======================================================")
        
        if not os.path.exists(self.tracker_file):
            print(f" No '{self.tracker_file}' found.")
            return

        df = pd.read_csv(self.tracker_file)
        pending = df[df['Result'] == 'PENDING']
        
        if pending.empty:
            print(" All bets are already graded!")
            self._print_tracker_summary(df)
            return

        print(f" Found {len(pending)} PENDING bets. Fetching actual box scores...")
        
        min_date = pd.to_datetime(pending['Date']).min()
        season_str = self.get_season_string(min_date)
        
        log = self.fetch_actuals(season_str)
        if log.empty:
            return

        log['PLAYER_LOWER'] = log['PLAYERNAME'].str.lower()
        df['PLAYER_LOWER'] = df['Player'].str.lower()
        
        graded_count = 0
        for idx, row in df.iterrows():
            if row['Result'] != 'PENDING':
                continue
                
            p_name = row['PLAYER_LOWER']
            b_date = pd.to_datetime(row['Date']).strftime('%Y-%m-%d')
            
            # Find matching game
            match = log[(log['PLAYER_LOWER'] == p_name) & (log['GAMEDATE'] == b_date)]
            if not match.empty:
                actual_pts = match.iloc[0]['PTS']
                df.at[idx, 'Actual_PTS'] = actual_pts
                
                v_line = float(row['Vegas_Line'])
                pick = row['Pick']
                
                if (pick == 'OVER' and actual_pts > v_line) or (pick == 'UNDER' and actual_pts < v_line):
                    df.at[idx, 'Result'] = 'WIN'
                elif actual_pts == v_line:
                    df.at[idx, 'Result'] = 'PUSH'
                else:
                    df.at[idx, 'Result'] = 'LOSS'
                graded_count += 1

        df = df.drop(columns=['PLAYER_LOWER'])
        df.to_csv(self.tracker_file, index=False)
        print(f" Successfully graded {graded_count} bets!")
        self._print_tracker_summary(df)

    def _print_tracker_summary(self, df):
        graded = df[df['Result'].isin(['WIN', 'LOSS', 'PUSH'])]
        wins = len(graded[graded['Result'] == 'WIN'])
        pushes = len(graded[graded['Result'] == 'PUSH'])
        losses = len(graded[graded['Result'] == 'LOSS'])
        
        actionable = wins + losses
        win_rate = (wins / actionable * 100) if actionable > 0 else 0.0
        profit = (wins * 90.90) - (losses * 100)
        
        five_stars = graded[graded['Stars'] == 5]
        fs_wins = len(five_stars[five_stars['Result'] == 'WIN'])
        fs_losses = len(five_stars[five_stars['Result'] == 'LOSS'])
        fs_rate = (fs_wins / (fs_wins + fs_losses) * 100) if (fs_wins + fs_losses) > 0 else 0.0

        print("\n=======================================================")
        print("  LIVE SYNDICATE ROI REPORT")
        print("=======================================================")
        print(f"Total Bets Settled: {len(graded)} ({pushes} Pushes)")
        print(f"Record:             {wins}W - {losses}L")
        print(f"Win Rate:           {win_rate:.1f}% (Break-even is 52.38%)")
        print(f"Net Profit:         ${profit:.2f} (Assuming $100 units)")
        print("-------------------------------------------------------")
        print(f"5-Star Lock Record: {fs_wins}W - {fs_losses}L ({fs_rate:.1f}%)")
        print("=======================================================\n")

    def evaluate_historical(self, strategy="standard"):
        """Sweeps Mega_Slate and Sportsbooks_Lines to test strategies."""
        title = "STANDARD STRATEGY (Tail All 3+ Stars)"
        if strategy == "optimized":
            title = "HYBRID OPTIMIZED STRATEGY (Fade 3/4 Stars, Tail 5 Stars)"
            
        print(f"=======================================================")
        print(f"  INITIALIZING MASS HISTORICAL BACKTEST")
        print(f"  STRATEGY: {title}")
        print(f"=======================================================")
        
        csv_files = glob.glob(os.path.join(self.slate_dir, 'Master_Slate_*.csv'))
        if not csv_files:
            print(f" No Master Slates found in {self.slate_dir}.")
            return
            
        print(f" Found {len(csv_files)} historical slates. Fetching ENTIRE SEASON box scores...")
        season_str = self.get_season_string(datetime.now())
        full_log = self.fetch_actuals(season_str)
        
        if full_log.empty: return

        master_bets = []

        for file in csv_files:
            date_str = os.path.basename(file).replace('Master_Slate_', '').replace('.csv', '')
            target_date = pd.to_datetime(date_str)
            
            try:
                df = pd.read_csv(file)
            except: continue

            vegas_file = os.path.join(self.archive_dir, f'vegas_props_{date_str}.json')
            if not os.path.exists(vegas_file):
                if os.path.exists('vegas_props.json') and date_str == datetime.now().strftime('%Y-%m-%d'):
                    vegas_file = 'vegas_props.json'
                else: continue

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
            
            api_date_str = target_date.strftime('%Y-%m-%d')
            daily_log = full_log[full_log['GAMEDATE'] == api_date_str].copy() if 'GAMEDATE' in full_log.columns else full_log.copy()

            daily_log['PLAYER_LOWER'] = daily_log['PLAYERNAME'].str.lower()
            df['PLAYER_LOWER'] = df['Player'].str.lower()
            
            merged = pd.merge(df, daily_log[['PLAYER_LOWER', 'PTS', 'MIN']], on='PLAYER_LOWER', how='inner')
            merged['MIN_FLOAT'] = merged['MIN'].apply(self.convert_minutes)
            merged = merged[merged['MIN_FLOAT'] >= 5.0].copy()

            for _, row in merged.iterrows():
                pred = row['Predicted_PTS']
                v_line = row['Vegas_Line']
                actual = row['PTS']
                edge = pred - v_line
                
                stars, reason = self.calculate_confidence(pred, row['Floor'], row['Ceiling'], v_line)
                
                if stars >= 3:
                    if strategy == "standard":
                        pick = "OVER" if edge > 0 else "UNDER"
                    elif strategy == "optimized":
                        if stars == 5:
                            pick = "OVER" if edge > 0 else "UNDER"
                        else:
                            pick = "UNDER" if edge > 0 else "OVER"
                            
                    result = 'LOSS'
                    if (pick == 'OVER' and actual > v_line) or (pick == 'UNDER' and actual < v_line):
                        result = 'WIN'
                    elif actual == v_line:
                        result = 'PUSH'
                        
                    master_bets.append({
                        'Player': row['Player'],
                        'Date': date_str,
                        'Stars': stars,
                        'Pick': pick,
                        'Result': result
                    })

        if not master_bets:
            print(" Could not find actionable bets to evaluate.")
            return

        bets_df = pd.DataFrame(master_bets)
        
        wins = len(bets_df[bets_df['Result'] == 'WIN'])
        pushes = len(bets_df[bets_df['Result'] == 'PUSH'])
        losses = len(bets_df[bets_df['Result'] == 'LOSS'])
        actionable = wins + losses
        win_rate = (wins / actionable * 100) if actionable > 0 else 0.0
        profit = (wins * 90.90) - (losses * 100)

        print("\n=======================================================")
        print(f"  HYPOTHETICAL PERFORMANCE REPORT")
        print("=======================================================")
        print(f"Total Bets Placed:  {len(bets_df)} ({pushes} Pushes)")
        print(f"Record:             {wins}W - {losses}L")
        print(f"Win Rate:           {win_rate:.1f}%  (Break-even is 52.38%)")
        print(f"Net Profit:         ${profit:.2f}")
        print("-------------------------------------------------------")
        print("  BREAKDOWN BY STAR RATING")
        print("-------------------------------------------------------")
        
        for star_tier in [5, 4, 3]:
            tier_df = bets_df[bets_df['Stars'] == star_tier]
            t_wins = len(tier_df[tier_df['Result'] == 'WIN'])
            t_losses = len(tier_df[tier_df['Result'] == 'LOSS'])
            t_rate = (t_wins / (t_wins + t_losses) * 100) if (t_wins + t_losses) > 0 else 0.0
            
            t_profit = (t_wins * 90.90) - (t_losses * 100)
            color_profit = f"+${t_profit:.2f}" if t_profit > 0 else f"-${abs(t_profit):.2f}"
            
            print(f" {star_tier}-Star Bets: {t_wins}W - {t_losses}L ({t_rate:.1f}%) | {color_profit}")
            
        print("=======================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Bet Tracker or Evaluate Historical Strategies.")
    parser.add_argument('--grade', action='store_true', help="Grade recent pending bets in your daily bet_tracker.csv")
    parser.add_argument('--gradeall', action='store_true', help="Sweep history using STANDARD Strategy (Tail all 3+ stars)")
    parser.add_argument('--optimized', action='store_true', help="Sweep history using HYBRID Strategy (Fade 3/4 stars, Tail 5 stars)")
    args = parser.parse_args()
    
    analyzer = BetAnalyzer()
    
    if args.grade:
        analyzer.grade_tracker()
    elif args.gradeall:
        analyzer.evaluate_historical(strategy="standard")
    elif args.optimized:
        analyzer.evaluate_historical(strategy="optimized")
    else:
        print("Please specify an action: --grade, --gradeall, or --optimized")
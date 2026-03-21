import pandas as pd
import json
import os
import argparse
import unicodedata
from datetime import datetime, timedelta
from nba_api.stats.endpoints import leaguegamelog

class BetAnalyzer:
    def __init__(self):
        self.tracker_file = 'bet_tracker.csv'
        self.vegas_file = 'vegas_props.json'

    def _normalize_name(self, name):
        """Strips accents, punctuation, and suffixes for exact comparisons"""
        name = ''.join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')
        name = name.lower().replace('.', '').replace('-', ' ').replace("'", "")
        name = name.replace(' jr', '').replace(' sr', '').replace(' iii', '').replace(' ii', '')
        return name.strip()

    def calculate_confidence(self, pred, floor, ceiling, line):
        """
        Uses the LightGBM Quantile Regression to grade the safety of the bet.
        """
        edge = pred - line
        
        if edge > 0:
            if floor > line: return 5, "⭐⭐⭐⭐⭐ (Floor clears Line)"
            elif pred >= line + 3.0: return 4, "⭐⭐⭐⭐ (Massive Median Edge)"
            elif pred >= line + 1.5: return 3, "⭐⭐⭐ (Solid Edge)"
            else: return 2, "⭐⭐ (Lean)"
        else:
            if ceiling < line: return 5, "⭐⭐⭐⭐⭐ (Ceiling under Line)"
            elif pred <= line - 3.0: return 4, "⭐⭐⭐⭐ (Massive Median Edge)"
            elif pred <= line - 1.5: return 3, "⭐⭐⭐ (Solid Edge)"
            else: return 2, "⭐⭐ (Lean)"

    def scan_for_bets(self, prediction_csv):
        print(f"🔍 Scanning {prediction_csv} against Vegas Lines...")
        
        if not os.path.exists(self.vegas_file):
            print(f"❌ '{self.vegas_file}' not found. Run odds_fetcher.py first!")
            return

        with open(self.vegas_file, 'r') as f:
            vegas_data = json.load(f).get('lines', {})

        df = pd.read_csv(prediction_csv)
        
        required_cols = ['Player', 'Predicted_PTS', 'Floor', 'Ceiling']
        for col in required_cols:
            if col not in df.columns:
                print(f"❌ CSV is missing required column: {col}")
                return

        today_str = datetime.now().strftime('%Y-%m-%d')
        new_bets = []

        print("\n===========================================================================")
        print(" 🎯 TOP PROP BETTING EDGES DISCOVERED")
        print("===========================================================================")
        print(f"{'Player':<22} | {'Vegas':<5} | {'AI Pred':<7} | {'Edge':<5} | {'Pick':<5} | {'Confidence'}")
        print("-" * 75)

        for _, row in df.iterrows():
            player = row['Player']
            pred = float(row['Predicted_PTS'])
            floor = float(row['Floor'])
            ceil = float(row['Ceiling'])
            
            v_line = vegas_data.get(player)
            if v_line is None:
                clean_player = self._normalize_name(player)
                p_parts = clean_player.split()
                
                for v_name, line in vegas_data.items():
                    clean_v_name = self._normalize_name(v_name)
                    v_parts = clean_v_name.split()
                    
                    if clean_player == clean_v_name:
                        v_line = line
                        break
                        
                    if len(p_parts) >= 2 and len(v_parts) >= 2:
                        if p_parts[-1] == v_parts[-1]:
                            f1 = p_parts[0]
                            f2 = v_parts[0]
                            if f1 == f2 or (len(f1) >= 3 and f1 in f2) or (len(f2) >= 3 and f2 in f1):
                                v_line = line
                                break
            
            if v_line is not None:
                edge = pred - v_line
                pick = "OVER" if edge > 0 else "UNDER"
                stars, reason = self.calculate_confidence(pred, floor, ceil, v_line)
                
                if stars >= 3:
                    color_edge = f"+{edge:.1f}" if edge > 0 else f"{edge:.1f}"
                    print(f"{player:<22} | {v_line:<5.1f} | {pred:<7.1f} | {color_edge:<5} | {pick:<5} | {reason}")
                    
                    new_bets.append({
                        "Date": today_str,
                        "Player": player,
                        "Vegas_Line": v_line,
                        "AI_Pred": pred,
                        "AI_Floor": floor,
                        "AI_Ceiling": ceil,
                        "Pick": pick,
                        "Edge": round(edge, 1),
                        "Stars": stars,
                        "Actual_PTS": "PENDING",
                        "Result": "PENDING"
                    })

        print("===========================================================================\n")

        if new_bets:
            tracker_df = pd.DataFrame(new_bets)
            if os.path.exists(self.tracker_file):
                existing_df = pd.read_csv(self.tracker_file)
                existing_df = existing_df[~((existing_df['Date'] == today_str) & (existing_df['Player'].isin([b['Player'] for b in new_bets])))]
                tracker_df = pd.concat([existing_df, tracker_df], ignore_index=True)
                
            tracker_df.to_csv(self.tracker_file, index=False)
            print(f"📝 Logged {len(new_bets)} premium bets to '{self.tracker_file}'.")
        else:
            print("🛑 No 3-Star+ edges found. DO NOT FORCE BETS.")

    def grade_pending_bets(self):
        if not os.path.exists(self.tracker_file):
            print(f"❌ No bet tracker found.")
            return

        df = pd.read_csv(self.tracker_file)
        pending_df = df[df['Result'] == 'PENDING'].copy()
        
        if pending_df.empty:
            print("✅ All logged bets have already been graded.")
            self._print_roi_report(df)
            return

        print(f"📊 Found {len(pending_df)} PENDING bets. Fetching actual box scores...")
        
        year = datetime.now().year
        season_str = f"{year-1}-{str(year)[-2:]}"
        try:
            log = leaguegamelog.LeagueGameLog(player_or_team_abbreviation='P', season=season_str).get_data_frames()[0]
            log.columns = [str(c).upper() for c in log.columns]
        except Exception as e:
            print(f"❌ Failed to connect to NBA API: {e}")
            return

        graded_count = 0
        
        for idx, row in pending_df.iterrows():
            player = row['Player']
            bet_date = row['Date']
            
            player_logs = log[log['PLAYER_NAME'] == player].copy()
            player_logs['GAME_DATE'] = pd.to_datetime(player_logs['GAME_DATE'])
            
            target_date = pd.to_datetime(bet_date)
            match = player_logs[player_logs['GAME_DATE'] >= target_date]
            
            if not match.empty:
                actual_pts = match.iloc[0]['PTS']
                df.at[idx, 'Actual_PTS'] = actual_pts
                
                line = float(row['Vegas_Line'])
                pick = row['Pick']
                
                if actual_pts == line:
                    df.at[idx, 'Result'] = 'PUSH'
                elif (pick == 'OVER' and actual_pts > line) or (pick == 'UNDER' and actual_pts < line):
                    df.at[idx, 'Result'] = 'WIN'
                else:
                    df.at[idx, 'Result'] = 'LOSS'
                    
                graded_count += 1

        df.to_csv(self.tracker_file, index=False)
        print(f"✅ Successfully graded {graded_count} bets!\n")
        self._print_roi_report(df)

    def _print_roi_report(self, df):
        completed = df[df['Result'].isin(['WIN', 'LOSS'])]
        if completed.empty: return
            
        wins = len(completed[completed['Result'] == 'WIN'])
        losses = len(completed[completed['Result'] == 'LOSS'])
        pushes = len(df[df['Result'] == 'PUSH'])
        total = wins + losses
        win_rate = (wins / total) * 100 if total > 0 else 0
        
        profit = (wins * 90.90) - (losses * 100)
        
        print("=======================================================")
        print(" 🏦 AI BETTING SYNDICATE ROI REPORT")
        print("=======================================================")
        print(f"Total Bets Settled: {total} ({pushes} Pushes)")
        print(f"Record:             {wins}W - {losses}L")
        print(f"Win Rate:           {win_rate:.1f}% (Break-even is 52.38%)")
        print(f"Net Profit:         ${profit:,.2f} (Assuming $100 units)")
        
        five_stars = completed[completed['Stars'] == 5]
        if not five_stars.empty:
            fw = len(five_stars[five_stars['Result'] == 'WIN'])
            fl = len(five_stars[five_stars['Result'] == 'LOSS'])
            f_rate = (fw / (fw+fl)) * 100 if (fw+fl) > 0 else 0
            print(f"5-Star Lock Record: {fw}W - {fl}L ({f_rate:.1f}%)")
        print("=======================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Prop Betting Analyzer")
    parser.add_step_group = parser.add_mutually_exclusive_group(required=True)
    parser.add_step_group.add_argument('--scan', type=str, help='Path to prediction CSV')
    parser.add_step_group.add_argument('--grade', action='store_true', help='Grade pending bets')
    
    args = parser.parse_args()
    analyzer = BetAnalyzer()
    
    if args.scan: analyzer.scan_for_bets(args.scan)
    elif args.grade: analyzer.grade_pending_bets()
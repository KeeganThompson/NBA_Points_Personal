import pandas as pd
import pulp
import math

class LineupOptimizer:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.df = pd.read_csv(csv_file)
        self.prepare_data()

    def mock_draftkings_salary(self, avg_pts):
        """
        Generates a realistic DraftKings salary. 
        Usually, DFS prices players at roughly $300 per projected fantasy point, 
        plus a baseline floor for active players.
        """
        if pd.isna(avg_pts) or avg_pts <= 0:
            return 3000
        
        base_salary = (avg_pts * 250) + 1500
        salary = int(math.ceil(base_salary / 100.0)) * 100
        
        return max(3000, min(salary, 12500))

    def mock_positions(self):
        """
        Assigns G, F, or C based on standard height/name assumptions since 
        the backtest CSV drops the position column. 
        (If feeding direct from app.py later, we can map true positions).
        """
        positions = []
        for i, row in self.df.iterrows():
            if i % 5 == 0: positions.append('C')
            elif i % 2 == 0: positions.append('F')
            else: positions.append('G')
        return positions

    def prepare_data(self):
        print("📊 Loading Player Pool and generating salaries...")
        
        self.df = self.df.dropna(subset=['Predicted', '10_Game_Avg'])
        
        self.df['Salary'] = self.df['10_Game_Avg'].apply(self.mock_draftkings_salary)
        
        self.df['Pos'] = self.mock_positions()
        
        self.df['Value_Rating'] = (self.df['Predicted'] / self.df['Salary']) * 1000
        
        self.df = self.df.sort_values('Value_Rating', ascending=False).reset_index(drop=True)

    def optimize(self, strategy="median"):
        """
        Runs Linear Programming to find the mathematically perfect 8-man roster.
        DraftKings Rules: 8 Players, Max Salary $50,000.
        Positional Needs (Adapted for our G/F/C format):
        - Minimum 3 Guards (PG, SG, G)
        - Minimum 3 Forwards (SF, PF, F)
        - Minimum 1 Center (C)
        - 1 UTIL (Any)
        """
        print(f"\n🧠 Running PuLP Linear Optimizer (Strategy: {strategy.upper()})...")
        
        prob = pulp.LpProblem("DFS_Optimal_Lineup", pulp.LpMaximize)
        
        num_players = len(self.df)
        
        player_vars = [pulp.LpVariable(f"player_{i}", cat="Binary") for i in range(num_players)]
        
        if strategy == "ceiling":
            target_metric = self.df['Ceiling'].tolist()
        else:
            target_metric = self.df['Predicted'].tolist()
            
        prob += pulp.lpSum([target_metric[i] * player_vars[i] for i in range(num_players)])
        
        salaries = self.df['Salary'].tolist()
        positions = self.df['Pos'].tolist()
        
        prob += pulp.lpSum([player_vars[i] for i in range(num_players)]) == 8
        
        prob += pulp.lpSum([salaries[i] * player_vars[i] for i in range(num_players)]) <= 50000
        
        is_guard = [1 if pos == 'G' else 0 for pos in positions]
        is_forward = [1 if pos == 'F' else 0 for pos in positions]
        is_center = [1 if pos == 'C' else 0 for pos in positions]
        
        prob += pulp.lpSum([is_guard[i] * player_vars[i] for i in range(num_players)]) >= 3
        prob += pulp.lpSum([is_forward[i] * player_vars[i] for i in range(num_players)]) >= 3
        prob += pulp.lpSum([is_center[i] * player_vars[i] for i in range(num_players)]) >= 1
        
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        
        if pulp.LpStatus[prob.status] != 'Optimal':
            print("❌ Could not find a valid lineup under the constraints.")
            return
            
        lineup = []
        total_salary = 0
        total_proj = 0
        
        for i in range(num_players):
            if player_vars[i].varValue == 1.0:
                p = self.df.iloc[i]
                lineup.append({
                    "Player": p['Player'],
                    "Pos": p['Pos'],
                    "Salary": p['Salary'],
                    "Prediction": p['Predicted'],
                    "Ceiling": p.get('Ceiling', 0.0),
                    "Value": p['Value_Rating']
                })
                total_salary += p['Salary']
                total_proj += p['Predicted'] if strategy == "median" else p['Ceiling']
                
        pos_order = {'G': 1, 'F': 2, 'C': 3}
        lineup.sort(key=lambda x: (pos_order.get(x['Pos'], 4), -x['Salary']))
        
        print("\n=======================================================")
        print(" 🏆 OPTIMAL DRAFTKINGS LINEUP GENERATED")
        print("=======================================================")
        print(f"{'Position':<6} | {'Player':<22} | {'Salary':<7} | {'Proj PTS':<8} | {'Value (PTS/$K)'}")
        print("-" * 65)
        for p in lineup:
            target_pts = p['Prediction'] if strategy == "median" else p['Ceiling']
            print(f"{p['Pos']:<6} | {p['Player']:<22} | ${p['Salary']:<6} | {target_pts:<8.1f} | {p['Value']:<4.2f}x")
        print("=======================================================")
        print(f"💰 Total Salary Used: ${total_salary:,} / $50,000")
        print(f"🎯 Total Projected:   {total_proj:.1f} PTS")
        print("=======================================================\n")

if __name__ == "__main__":
    try:
        opt = LineupOptimizer("Backtest_Results.csv")
        
        opt.optimize(strategy="median")
        
        opt.optimize(strategy="ceiling")
        
    except FileNotFoundError:
        print("❌ Could not find 'Backtest_Results.csv'. Run your backtester or web app first to generate data!")
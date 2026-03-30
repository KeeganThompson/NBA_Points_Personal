import pandas as pd

class MinuteAllocator:
    def convert_minutes(self, x):
        if pd.isna(x): return 0.0
        if isinstance(x, (int, float)): return float(x)
        x_str = str(x).strip()
        try:
            if ":" in x_str:
                return float(x_str.split(":")[0]) + (float(x_str.split(":")[1]) / 60.0)
            return float(x_str)
        except: return 0.0

    def allocate(self, active_players, injured_players, starters, all_player_data):
        base_mins = {}
        
        for p in active_players:
            if p in all_player_data and not all_player_data[p].empty:
                df = all_player_data[p]
                if 'MIN' in df.columns:
                    recent_mins = [self.convert_minutes(m) for m in df['MIN'].tail(10)]
                    base_mins[p] = sum(recent_mins) / len(recent_mins) if recent_mins else 15.0
                else: base_mins[p] = 15.0
            else:
                base_mins[p] = 15.0

        for p in active_players:
            if p in starters and base_mins[p] < 26.0:
                base_mins[p] = 28.0 
            elif p not in starters and base_mins[p] > 28.0:
                base_mins[p] = 22.0 

        vacated_mins = 0.0
        for p in injured_players:
            if p in all_player_data and not all_player_data[p].empty:
                df = all_player_data[p]
                if 'MIN' in df.columns:
                    recent_mins = [self.convert_minutes(m) for m in df['MIN'].tail(10)]
                    if recent_mins: vacated_mins += sum(recent_mins) / len(recent_mins)

        if vacated_mins > 0:
            for p in active_players:
                if p in starters and base_mins[p] < 34.0:
                    bump = min(vacated_mins * 0.15, 36.0 - base_mins[p])
                    base_mins[p] += bump
                    vacated_mins -= bump
                elif p not in starters:
                    bump = min(vacated_mins * 0.20, 26.0 - base_mins[p])
                    if bump > 0:
                        base_mins[p] += bump
                        vacated_mins -= bump

        total_allocated = sum(base_mins.values())
        if total_allocated > 0:
            scale = 240.0 / total_allocated
            scale = max(0.85, min(scale, 1.15))
            for p in base_mins:
                base_mins[p] *= scale
                if base_mins[p] > 40.0: base_mins[p] = 40.0 # Hard Cap
                if base_mins[p] < 0.0: base_mins[p] = 0.0

        return base_mins
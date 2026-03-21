import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb

class Predictor:
    def __init__(self):
        self.feature_cols = [
            'Home', 'Days_Rest', 'Return_From_Injury', 
            'Opp_DvP_Advantage', 'Opp_Pace', 'Blowout_Risk', 
            'L3_PTS', 'L5_PTS', 'L10_PTS', 
            'Proj_Minutes', 'L5_FGA', 'L5_USG',
            'Season_Avg_PTS',
            'Is_Rookie',
            'L5_PPM',
            'Trend_Multiplier',
            'USG_Delta',         
            'PPM_Delta',         
            'Is_Guard',          
            'Is_Forward',
            'Is_Center',
            'MIN_StdDev',        
            'PPM_StdDev',
            'Games_In_7_Days' 
        ]

    def convert_minutes(self, x):
        if pd.isna(x): return 0.0
        if isinstance(x, (int, float)): return float(x)
        
        x_str = str(x).strip()
        try:
            if ":" in x_str:
                parts = x_str.split(":")
                return float(parts[0]) + (float(parts[1]) / 60.0)
            return float(x_str)
        except:
            return 0.0

    def prepare_data(self, player_df, adv_stats, team_map, current_team_id, experience, position, dvp_ranks):
        df = player_df.copy()
        df['MP_Float'] = df['MP'].apply(self.convert_minutes)
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        
        df['Is_Rookie'] = 1 if experience == 0 else 0
        df['Is_Guard'] = 1 if position == 'G' else 0
        df['Is_Forward'] = 1 if position == 'F' else 0
        df['Is_Center'] = 1 if position == 'C' else 0
        
        df['PPM'] = df['PTS'] / (df['MP_Float'] + 0.1)
        df['Days_Rest'] = df['GAME_DATE'].diff().dt.days.fillna(3)
        
        recovery_values = []
        current_recovery = 0.0
        for rest in df['Days_Rest']:
            if rest >= 7: current_recovery = 1.0 
            elif current_recovery > 0: current_recovery = max(0.0, current_recovery - 0.2)
            recovery_values.append(current_recovery)
            
        df['Return_From_Injury'] = recovery_values
        df['Days_Rest'] = df['Days_Rest'].clip(upper=7)

        df['Opp_ID'] = df['Opp'].map(team_map)
        df = df.merge(adv_stats, left_on='Opp_ID', right_on='TEAM_ID', how='left')
        
        df = df.sort_values('GAME_DATE').reset_index(drop=True)
        
        df_idx = df.set_index('GAME_DATE').sort_index()
        df['Games_In_7_Days'] = df_idx['PTS'].rolling('7D').count().values - 1
        
        rename_map = {'PACE': 'Opp_Pace', 'NET_RATING': 'Opp_Net_Rating'}
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        
        if 'Opp_Pace' not in df.columns: df['Opp_Pace'] = 100.0
        if 'Opp_Net_Rating' not in df.columns: df['Opp_Net_Rating'] = 0.0
        df['Opp_Pace'] = df['Opp_Pace'].fillna(100.0)
        df['Opp_Net_Rating'] = df['Opp_Net_Rating'].fillna(0.0)

        df['L5_MIN'] = df['MP_Float'].rolling(window=5, min_periods=1).mean().shift(1)
        df['Proj_Minutes'] = df['L5_MIN'] 
        df['L5_FGA'] = df['FGA'].rolling(window=5, min_periods=1).mean().shift(1)
        df['L5_USG'] = (df['FGA'].rolling(5).sum().shift(1)) / (df['MP_Float'].rolling(5).sum().shift(1) + 0.1)

        df['MIN_StdDev'] = df['MP_Float'].rolling(window=10, min_periods=1).std().shift(1).fillna(0.0)
        df['PPM_StdDev'] = df['PPM'].rolling(window=10, min_periods=1).std().shift(1).fillna(0.0)

        advantages = []
        for _, row in df.iterrows():
            opp_str = str(row['Opp'])
            rank = dvp_ranks.get(opp_str, {}).get(position, 15.5)
            usg = row['L5_USG'] if pd.notna(row['L5_USG']) else 0.20
            adv = (rank - 15.5) * (usg / 0.20)
            advantages.append(adv)
        df['Opp_DvP_Advantage'] = advantages

        team_net_val = adv_stats.loc[adv_stats['TEAM_ID'] == current_team_id, 'NET_RATING'] if 'NET_RATING' in adv_stats.columns else pd.Series([0.0])
        team_net_val = team_net_val.iloc[0] if not team_net_val.empty else 0.0
        df['Blowout_Risk'] = (team_net_val - df['Opp_Net_Rating']).abs()

        df['L5_PPM'] = df['PPM'].rolling(window=5, min_periods=1).mean().shift(1)
        df['L3_PTS'] = df['PTS'].rolling(window=3, min_periods=1).mean().shift(1)
        df['L5_PTS'] = df['PTS'].rolling(window=5, min_periods=1).mean().shift(1)
        df['L10_PTS'] = df['PTS'].rolling(window=10, min_periods=1).mean().shift(1)
        
        df['L3_USG'] = (df['FGA'].rolling(3).sum().shift(1)) / (df['MP_Float'].rolling(3).sum().shift(1) + 0.1)
        df['L10_USG'] = (df['FGA'].rolling(10).sum().shift(1)) / (df['MP_Float'].rolling(10).sum().shift(1) + 0.1)
        df['USG_Delta'] = (df['L3_USG'] - df['L10_USG']).clip(-0.15, 0.15)

        df['L3_PPM'] = df['PPM'].rolling(3, min_periods=1).mean().shift(1)
        df['L10_PPM'] = df['PPM'].rolling(10, min_periods=1).mean().shift(1)
        df['PPM_Delta'] = (df['L3_PPM'] - df['L10_PPM']).clip(-0.25, 0.25)
        
        df['Season_Avg_PTS'] = df['PTS'].expanding().mean().shift(1)
        df['Trend_Multiplier'] = df['L5_PTS'] / (df['Season_Avg_PTS'] + 0.1)

        df = df.bfill() 
        df = df.fillna(0)

        return df

    def predict_next_game(self, player_df, adv_stats, team_map, current_team_id, next_game_data, experience, position, dvp_ranks, is_starter):
        target_opp_str = next_game_data.get('Opp')
        target_opp_id = next_game_data.get('Opp_ID')
        
        engineered_df = self.prepare_data(player_df, adv_stats, team_map, current_team_id, experience, position, dvp_ranks)
        
        X = engineered_df[self.feature_cols].to_numpy()
        y = engineered_df['PTS'].to_numpy()
        
        n_samples = len(engineered_df)
        if experience == 0: weights = np.linspace(0.05, 5.0, n_samples) 
        else: weights = np.linspace(0.2, 1.8, n_samples) 
            
        next_date = pd.to_datetime(next_game_data['Date'])
        last_game_date = pd.to_datetime(player_df.iloc[-1]['GAME_DATE'])
        next_game_rest = (next_date - last_game_date).days
        
        last_recovery_val = engineered_df.iloc[-1]['Return_From_Injury']
        next_recovery_val = 1.0 if next_game_rest >= 7 else max(0.0, last_recovery_val - 0.2)
        
        target_stats = adv_stats[adv_stats['TEAM_ID'] == target_opp_id] if 'TEAM_ID' in adv_stats.columns else pd.DataFrame()
        target_pace_val = target_stats['PACE'].iloc[0] if not target_stats.empty and 'PACE' in target_stats.columns else 100.0
        target_net_val = target_stats['NET_RATING'].iloc[0] if not target_stats.empty and 'NET_RATING' in target_stats.columns else 0.0
        
        team_net_val = adv_stats.loc[adv_stats['TEAM_ID'] == current_team_id, 'NET_RATING'] if 'TEAM_ID' in adv_stats.columns and 'NET_RATING' in adv_stats.columns else pd.Series([0.0])
        team_net_val = team_net_val.iloc[0] if not team_net_val.empty else 0.0
        
        next_blowout_risk = abs(team_net_val - target_net_val)
        
        current_l3_pts = player_df['PTS'].tail(3).mean()
        current_l5_pts = player_df['PTS'].tail(5).mean()
        current_l10_pts = player_df['PTS'].tail(10).mean()
        current_l5_min = player_df['MP'].apply(self.convert_minutes).tail(5).mean()
        current_l5_fga = player_df['FGA'].tail(5).mean()
        
        current_min_std = player_df['MP'].apply(self.convert_minutes).tail(10).std()
        current_min_std = 0.0 if pd.isna(current_min_std) else current_min_std
        
        ppm_series = player_df['PTS'] / (player_df['MP'].apply(self.convert_minutes) + 0.1)
        current_ppm_std = ppm_series.tail(10).std()
        current_ppm_std = 0.0 if pd.isna(current_ppm_std) else current_ppm_std
        
        min_sum_3 = player_df['MP'].apply(self.convert_minutes).tail(3).sum()
        min_sum_5 = player_df['MP'].apply(self.convert_minutes).tail(5).sum()
        min_sum_10 = player_df['MP'].apply(self.convert_minutes).tail(10).sum()
        
        current_l3_usg = player_df['FGA'].tail(3).sum() / (min_sum_3 + 0.1)
        current_l5_usg = player_df['FGA'].tail(5).sum() / (min_sum_5 + 0.1)
        current_l10_usg = player_df['FGA'].tail(10).sum() / (min_sum_10 + 0.1)
        current_usg_delta = np.clip(current_l3_usg - current_l10_usg, -0.15, 0.15)
        current_ppm_delta = np.clip((player_df['PTS'].tail(3).sum() / (min_sum_3 + 0.1)) - (player_df['PTS'].tail(10).sum() / (min_sum_10 + 0.1)), -0.25, 0.25)

        current_season_avg = player_df['PTS'].mean()
        current_trend = current_l5_pts / (current_season_avg + 0.1)
        current_ppm = player_df['PTS'].tail(5).sum() / (min_sum_5 + 0.1)
        
        opp_dvp_all = dvp_ranks.get(target_opp_str, {'G': 15.5, 'F': 15.5, 'C': 15.5})
        target_dvp_rank = opp_dvp_all.get(position, 15.5)
        
        best_matchup_rank = max(opp_dvp_all.values())
        worst_matchup_rank = min(opp_dvp_all.values())
        
        is_alpha_target = (target_dvp_rank == best_matchup_rank) and (target_dvp_rank >= 18.0)
        is_avoid_target = (target_dvp_rank == worst_matchup_rank) and (target_dvp_rank <= 12.0)

        safe_usg = min(current_l5_usg, 1.0)
        target_dvp_advantage = (target_dvp_rank - 15.5) * (safe_usg / 0.20)

        proj_minutes = current_l5_min
        if not is_starter and proj_minutes < 4.0:
            return {"prediction": 0.0, "floor": 0.0, "ceiling": 0.0}

        if is_starter and current_l5_min < 24.0: proj_minutes = 26.0  
        elif not is_starter and current_l5_min > 28.0: proj_minutes = current_l5_min * 0.70 
            
        safe_l10_ppm = (player_df['PTS'].tail(10).sum() / (min_sum_10 + 0.1))
        if pd.isna(safe_l10_ppm) or safe_l10_ppm == 0: safe_l10_ppm = current_season_avg / (current_l5_min+0.1)
        
        dynamic_base = proj_minutes * safe_l10_ppm
        if pd.isna(dynamic_base) or dynamic_base <= 0: dynamic_base = 0.5 

        games_in_7 = next_game_data.get('Games_In_7_Days', 2.0)

        next_game_features = np.array([[
            next_game_data['Home'],                          
            min(next_game_rest, 7),                          
            next_recovery_val,                               
            target_dvp_advantage,                                  
            target_pace_val,         
            min(next_blowout_risk, 20.0),       
            current_l3_pts,                                  
            current_l5_pts,                                  
            current_l10_pts,                                 
            proj_minutes,                                
            current_l5_fga,                                  
            current_l5_usg,                                  
            current_season_avg,
            1 if experience == 0 else 0,
            current_ppm,
            current_trend,
            current_usg_delta,
            current_ppm_delta,
            1 if position == 'G' else 0,
            1 if position == 'F' else 0,
            1 if position == 'C' else 0,
            current_min_std,
            current_ppm_std,
            games_in_7 
        ]])

        xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror', random_state=42, n_estimators=50, 
            learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8, 
            min_child_weight=3, reg_alpha=1.0, reg_lambda=1.5, base_score=dynamic_base, n_jobs=-1
        )
        xgb_model.fit(X, y, sample_weight=weights)
        xgb_pred = xgb_model.predict(next_game_features)[0]
        
        lgb_model = lgb.LGBMRegressor(
            random_state=42, n_estimators=50, learning_rate=0.05, max_depth=3, 
            subsample=0.8, min_child_samples=3, reg_alpha=1.0, reg_lambda=1.5, 
            n_jobs=-1, verbose=-1
        )
        lgb_model.fit(X, y, sample_weight=weights)
        lgb_pred = lgb_model.predict(next_game_features)[0]
        
        raw_prediction = (xgb_pred + lgb_pred) / 2.0
        
        lgb_floor = lgb.LGBMRegressor(
            objective='quantile', alpha=0.2, random_state=42, n_estimators=40, 
            learning_rate=0.05, max_depth=3, verbose=-1
        )
        lgb_floor.fit(X, y, sample_weight=weights)
        raw_floor = lgb_floor.predict(next_game_features)[0]
        
        lgb_ceil = lgb.LGBMRegressor(
            objective='quantile', alpha=0.8, random_state=42, n_estimators=40, 
            learning_rate=0.05, max_depth=3, verbose=-1
        )
        lgb_ceil.fit(X, y, sample_weight=weights)
        raw_ceil = lgb_ceil.predict(next_game_features)[0]

        multiplier = 1.0
        if is_alpha_target:
            multiplier *= 1.18 if safe_usg >= 0.20 else 1.08
        if is_avoid_target:
            multiplier *= 0.85 if safe_usg >= 0.20 else 0.80
            
        if next_blowout_risk >= 16.0:
            if current_l5_min >= 30.0: multiplier *= 0.82  
            elif current_l5_min <= 15.0: multiplier *= 1.18  
        elif next_blowout_risk >= 10.0:
            if current_l5_min >= 30.0: multiplier *= 0.90  
            elif current_l5_min <= 15.0: multiplier *= 1.10
                
        if games_in_7 >= 4.0:
            multiplier *= 0.95
        if next_game_rest <= 1:
            multiplier *= 0.92 if experience >= 5 else 0.96 
            
        prediction = raw_prediction * multiplier
        floor = raw_floor * multiplier
        ceiling = raw_ceil * multiplier
        
        safe_pts_base = current_l10_pts if pd.notna(current_l10_pts) and current_l10_pts > 0 else current_season_avg
        volatility_bonus = current_min_std * 0.8
        max_cap = (safe_pts_base * 1.60) + 10.0 + volatility_bonus
        min_cap = max(0.0, (safe_pts_base * 0.40) - 5.0)
        
        prediction = np.clip(prediction, min_cap, max_cap)
        floor = np.clip(floor, min_cap, prediction - 0.5) 
        ceiling = np.clip(ceiling, prediction + 0.5, max_cap)
            
        return {
            "prediction": float(prediction),
            "floor": float(floor),
            "ceiling": float(ceiling)
        }
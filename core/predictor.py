import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import optuna
import random
import warnings
from sklearn.metrics import mean_absolute_error

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

class Predictor:
    def __init__(self):
        self.feature_cols = [
            'Home', 'Days_Rest', 'Return_From_Injury', 
            'Opp_DvP_Advantage', 'Opp_Pace', 'Blowout_Risk', 
            'L3_PTS', 'L5_PTS', 'L10_PTS', 
            'Proj_Minutes', 'L5_FGA', 'L5_USG',
            'Season_Avg_PTS',
            'Location_Avg_PTS',    
            'Is_Rookie',
            'L5_PTS_Per_100',      
            'Trend_Multiplier',
            'USG_Delta',         
            'PTS_Per_100_Delta',   
            'Is_Guard',          
            'Is_Forward',
            'Is_Center',
            'MIN_StdDev',        
            'PTS_Per_100_StdDev',  
            'Games_In_7_Days',
            'Vacated_Team_PTS',
            'Opp_Def_Trend',
            'Opp_Def_PNR_PPP',     
            'Opp_Def_SpotUp_PPP',
            'Playtype_Advantage' 
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
        except: return 0.0

    def prepare_data(self, player_df, adv_stats, team_map, current_team_id, experience, position, dvp_ranks):
        df = player_df.copy()
        if df.empty:
            return df
            
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
        
        rename_map = {'PACE': 'Opp_Pace', 'NET_RATING': 'Opp_Net_Rating', 'DEF_TREND': 'Opp_Def_Trend'}
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        
        if 'Opp_Pace' not in df.columns: df['Opp_Pace'] = 100.0
        if 'Opp_Net_Rating' not in df.columns: df['Opp_Net_Rating'] = 0.0
        if 'Opp_Def_Trend' not in df.columns: df['Opp_Def_Trend'] = 0.0 
        
        if 'DEF_PNR_PPP' not in df.columns: df['DEF_PNR_PPP'] = 0.95
        if 'DEF_SPOTUP_PPP' not in df.columns: df['DEF_SPOTUP_PPP'] = 1.00
        
        df['Opp_Pace'] = df['Opp_Pace'].fillna(100.0)
        df['Opp_Net_Rating'] = df['Opp_Net_Rating'].fillna(0.0)
        df['Opp_Def_Trend'] = df['Opp_Def_Trend'].fillna(0.0)
        df['Opp_Def_PNR_PPP'] = df['DEF_PNR_PPP'].fillna(0.95)
        df['Opp_Def_SpotUp_PPP'] = df['DEF_SPOTUP_PPP'].fillna(1.00)

        df['Possessions'] = (df['MP_Float'] / 48.0) * df['Opp_Pace']
        df['PTS_Per_100'] = (df['PTS'] / (df['Possessions'] + 0.1)) * 100.0

        df['L5_MIN'] = df['MP_Float'].ewm(span=5, adjust=False).mean().shift(1)
        df['Proj_Minutes'] = df['L5_MIN'] 
        df['L5_FGA'] = df['FGA'].ewm(span=5, adjust=False).mean().shift(1)
        df['L5_USG'] = df['FGA'].ewm(span=5, adjust=False).mean().shift(1) / (df['MP_Float'].ewm(span=5, adjust=False).mean().shift(1) + 0.1)

        df['MIN_StdDev'] = df['MP_Float'].rolling(window=10, min_periods=1).std().shift(1).fillna(0.0)
        df['PTS_Per_100_StdDev'] = df['PTS_Per_100'].rolling(window=10, min_periods=1).std().shift(1).fillna(0.0)

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

        df['Season_Avg_PTS'] = df['PTS'].expanding().mean().shift(1)
        
        df['Home_PTS'] = np.where(df['Home'] == 1, df['PTS'], np.nan)
        df['Away_PTS'] = np.where(df['Home'] == 0, df['PTS'], np.nan)
        df['Home_Avg'] = df['Home_PTS'].expanding().mean().ffill().shift(1).fillna(df['Season_Avg_PTS'])
        df['Away_Avg'] = df['Away_PTS'].expanding().mean().ffill().shift(1).fillna(df['Season_Avg_PTS'])
        df['Location_Avg_PTS'] = np.where(df['Home'] == 1, df['Home_Avg'], df['Away_Avg'])
        df['Location_Avg_PTS'] = df['Location_Avg_PTS'].fillna(df['Season_Avg_PTS'])

        df['L5_PTS_Per_100'] = df['PTS_Per_100'].ewm(span=5, adjust=False).mean().shift(1)
        df['L3_PTS'] = df['PTS'].ewm(span=3, adjust=False).mean().shift(1)
        df['L5_PTS'] = df['PTS'].ewm(span=5, adjust=False).mean().shift(1)
        df['L10_PTS'] = df['PTS'].ewm(span=10, adjust=False).mean().shift(1)
        
        df['L3_USG'] = df['FGA'].ewm(span=3, adjust=False).mean().shift(1) / (df['MP_Float'].ewm(span=3, adjust=False).mean().shift(1) + 0.1)
        df['L10_USG'] = df['FGA'].ewm(span=10, adjust=False).mean().shift(1) / (df['MP_Float'].ewm(span=10, adjust=False).mean().shift(1) + 0.1)
        df['USG_Delta'] = (df['L3_USG'] - df['L10_USG']).clip(-0.15, 0.15)

        df['L3_PTS_Per_100'] = df['PTS_Per_100'].ewm(span=3, adjust=False).mean().shift(1)
        df['L10_PTS_Per_100'] = df['PTS_Per_100'].ewm(span=10, adjust=False).mean().shift(1)
        df['PTS_Per_100_Delta'] = (df['L3_PTS_Per_100'] - df['L10_PTS_Per_100']).clip(-15.0, 15.0)
        
        df['Trend_Multiplier'] = df['L5_PTS'] / (df['Season_Avg_PTS'] + 0.1)
        df['Vacated_Team_PTS'] = 0.0
        
        df['Playtype_Advantage'] = np.where(df['Is_Guard'] == 1, (df['Opp_Def_PNR_PPP'] - 0.95) * 6.0, (df['Opp_Def_SpotUp_PPP'] - 1.00) * 4.0)

        df = df.bfill().fillna(0)
        return df

    def predict_next_game(self, player_df, adv_stats, team_map, current_team_id, next_game_data, experience, position, dvp_ranks, is_starter, vacated_pts=0.0, playtype_delta=0.0):
        engineered_df = self.prepare_data(player_df, adv_stats, team_map, current_team_id, experience, position, dvp_ranks)
        
        if engineered_df.empty or len(engineered_df) < 5:
            safe_avg = player_df['PTS'].mean() if not player_df.empty else 0.0
            if pd.isna(safe_avg): safe_avg = 0.0
            return {
                "prediction": float(safe_avg),
                "floor": float(max(0.0, safe_avg - 3.0)),
                "ceiling": float(safe_avg + 4.0)
            }

        target_opp_str = next_game_data.get('Opp')
        target_opp_id = next_game_data.get('Opp_ID')
        
        X_df = engineered_df[self.feature_cols]
        X = X_df.to_numpy()
        y = engineered_df['PTS'].to_numpy()
        
        n_samples = len(engineered_df)
        if experience == 0: weights = np.linspace(0.1, 2.5, n_samples) 
        else: weights = np.linspace(0.3, 1.5, n_samples) 
            
        next_date = pd.to_datetime(next_game_data['Date'])
        last_game_date = pd.to_datetime(player_df.iloc[-1]['GAME_DATE'])
        next_game_rest = (next_date - last_game_date).days
        last_recovery_val = engineered_df.iloc[-1]['Return_From_Injury']
        next_recovery_val = 1.0 if next_game_rest >= 7 else max(0.0, last_recovery_val - 0.2)
        
        target_stats = adv_stats[adv_stats['TEAM_ID'] == target_opp_id] if 'TEAM_ID' in adv_stats.columns else pd.DataFrame()
        target_pace_val = target_stats['PACE'].iloc[0] if not target_stats.empty and 'PACE' in target_stats.columns else 100.0
        target_net_val = target_stats['NET_RATING'].iloc[0] if not target_stats.empty and 'NET_RATING' in target_stats.columns else 0.0
        target_def_trend = target_stats['DEF_TREND'].iloc[0] if not target_stats.empty and 'DEF_TREND' in target_stats.columns else 0.0
        
        target_pnr_ppp = target_stats['DEF_PNR_PPP'].iloc[0] if not target_stats.empty and 'DEF_PNR_PPP' in target_stats.columns else 0.95
        target_spotup_ppp = target_stats['DEF_SPOTUP_PPP'].iloc[0] if not target_stats.empty and 'DEF_SPOTUP_PPP' in target_stats.columns else 1.00
        
        team_net_val = adv_stats.loc[adv_stats['TEAM_ID'] == current_team_id, 'NET_RATING'] if 'TEAM_ID' in adv_stats.columns and 'NET_RATING' in adv_stats.columns else pd.Series([0.0])
        team_net_val = team_net_val.iloc[0] if not team_net_val.empty else 0.0
        next_blowout_risk = abs(team_net_val - target_net_val)
        
        current_l3_pts = engineered_df['PTS'].ewm(span=3, adjust=False).mean().iloc[-1]
        current_l5_pts = engineered_df['PTS'].ewm(span=5, adjust=False).mean().iloc[-1]
        current_l10_pts = engineered_df['PTS'].ewm(span=10, adjust=False).mean().iloc[-1]
        
        current_l5_min = engineered_df['MP_Float'].ewm(span=5, adjust=False).mean().iloc[-1]
        current_l5_fga = engineered_df['FGA'].ewm(span=5, adjust=False).mean().iloc[-1]
        
        current_min_std = engineered_df['MP_Float'].tail(10).std()
        current_min_std = 0.0 if pd.isna(current_min_std) else current_min_std
        
        current_pts_per_100_std = engineered_df['PTS_Per_100'].tail(10).std()
        current_pts_per_100_std = 0.0 if pd.isna(current_pts_per_100_std) else current_pts_per_100_std
        
        current_l3_usg = engineered_df['FGA'].ewm(span=3, adjust=False).mean().iloc[-1] / (engineered_df['MP_Float'].ewm(span=3, adjust=False).mean().iloc[-1] + 0.1)
        current_l5_usg = current_l5_fga / (current_l5_min + 0.1)
        current_l10_usg = engineered_df['FGA'].ewm(span=10, adjust=False).mean().iloc[-1] / (engineered_df['MP_Float'].ewm(span=10, adjust=False).mean().iloc[-1] + 0.1)
        current_usg_delta = np.clip(current_l3_usg - current_l10_usg, -0.15, 0.15)
        
        current_l5_pts_per_100 = engineered_df['PTS_Per_100'].ewm(span=5, adjust=False).mean().iloc[-1]
        current_l3_pts_per_100 = engineered_df['PTS_Per_100'].ewm(span=3, adjust=False).mean().iloc[-1]
        current_l10_pts_per_100 = engineered_df['PTS_Per_100'].ewm(span=10, adjust=False).mean().iloc[-1]
        current_pts_per_100_delta = np.clip(current_l3_pts_per_100 - current_l10_pts_per_100, -15.0, 15.0)

        current_season_avg = engineered_df['PTS'].mean()
        
        is_home_tonight = next_game_data['Home'] == 1
        loc_df = engineered_df[engineered_df['Home'] == (1 if is_home_tonight else 0)]
        current_location_avg = loc_df['PTS'].mean() if not loc_df.empty else current_season_avg
        if pd.isna(current_location_avg): current_location_avg = current_season_avg

        current_trend = current_l5_pts / (current_season_avg + 0.1)
        
        opp_dvp_all = dvp_ranks.get(target_opp_str, {'G': 15.5, 'F': 15.5, 'C': 15.5})
        target_dvp_rank = opp_dvp_all.get(position, 15.5)

        safe_usg = min(current_l5_usg, 1.0)
        target_dvp_advantage = (target_dvp_rank - 15.5) * (safe_usg / 0.20)

        proj_minutes = current_l5_min
        if not is_starter and proj_minutes < 4.0:
            return {"prediction": 0.0, "floor": 0.0, "ceiling": 0.0}

        if is_starter and current_l5_min < 24.0: proj_minutes = 26.0  
        elif not is_starter and current_l5_min > 28.0: proj_minutes = current_l5_min * 0.70 
            
        expected_possessions = (proj_minutes / 48.0) * target_pace_val
        safe_l10_per_100 = current_l10_pts_per_100
        if pd.isna(safe_l10_per_100) or safe_l10_per_100 == 0: 
            safe_l10_per_100 = (current_season_avg / (current_l5_min + 0.1)) * 100.0 * (48.0 / 100.0)
            
        dynamic_base = expected_possessions * (safe_l10_per_100 / 100.0)
        
        games_in_7 = next_game_data.get('Games_In_7_Days', 2.0)

        next_game_features = pd.DataFrame([[
            1 if is_home_tonight else 0, min(next_game_rest, 7), next_recovery_val,                               
            target_dvp_advantage, target_pace_val, min(next_blowout_risk, 20.0),       
            current_l3_pts, current_l5_pts, current_l10_pts, proj_minutes, current_l5_fga, current_l5_usg,                                  
            current_season_avg, current_location_avg, 1 if experience == 0 else 0,
            current_l5_pts_per_100, current_trend, current_usg_delta, current_pts_per_100_delta,
            1 if position == 'G' else 0, 1 if position == 'F' else 0, 1 if position == 'C' else 0,
            current_min_std, current_pts_per_100_std, games_in_7, float(vacated_pts),
            float(target_def_trend), float(target_pnr_ppp), float(target_spotup_ppp), float(playtype_delta) 
        ]], columns=self.feature_cols)

        lgb_model = lgb.LGBMRegressor(
            objective='huber', random_state=None, n_estimators=50, learning_rate=0.04,
            max_depth=3, min_child_samples=min(4, max(1, len(X) // 3)), verbose=-1
        )
        lgb_model.fit(X_df, y, sample_weight=weights)
        lgb_pred = lgb_model.predict(next_game_features)[0]

        xgb_model = xgb.XGBRegressor(
            objective='reg:absoluteerror', random_state=None, n_estimators=50, 
            learning_rate=0.03, max_depth=3, subsample=0.8, colsample_bytree=0.8, 
            min_child_weight=min(4, max(1, len(X) // 3)), base_score=dynamic_base, n_jobs=-1
        )
        xgb_model.fit(X_df, y, sample_weight=weights)
        xgb_pred = xgb_model.predict(next_game_features)[0]
        
        prediction = (xgb_pred + lgb_pred) / 2.0
        
        lgb_floor = lgb.LGBMRegressor(
            objective='quantile', alpha=0.15, random_state=None, n_estimators=40, 
            learning_rate=0.05, max_depth=2, verbose=-1, min_child_samples=min(4, max(1, len(X) // 3))
        )
        lgb_floor.fit(X_df, y, sample_weight=weights)
        floor = lgb_floor.predict(next_game_features)[0]
        
        lgb_ceil = lgb.LGBMRegressor(
            objective='quantile', alpha=0.85, random_state=None, n_estimators=40, 
            learning_rate=0.05, max_depth=2, verbose=-1, min_child_samples=min(4, max(1, len(X) // 3))
        )
        lgb_ceil.fit(X_df, y, sample_weight=weights)
        ceiling = lgb_ceil.predict(next_game_features)[0]

        multiplier = 1.0
        if vacated_pts >= 15.0:
            if is_starter: multiplier *= 1.15
            elif current_l5_usg >= 0.20: multiplier *= 1.10
            
        prediction *= multiplier
        floor *= multiplier
        ceiling *= multiplier
        
        safe_pts_base = current_l10_pts if pd.notna(current_l10_pts) and current_l10_pts > 0 else current_season_avg
        volatility_bonus = current_min_std * 0.75
        
        max_cap = (safe_pts_base * 1.50) + 6.0 + volatility_bonus
        min_cap = max(0.0, (safe_pts_base * 0.50) - 5.0)
        
        prediction = np.clip(prediction, min_cap, max_cap)
        floor = np.clip(floor, min_cap, prediction - 0.5) 
        ceiling = np.clip(ceiling, prediction + 0.5, max_cap)
            
        return {
            "prediction": float(prediction),
            "floor": float(floor),
            "ceiling": float(ceiling)
        }
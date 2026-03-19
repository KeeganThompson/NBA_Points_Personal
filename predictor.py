import pandas as pd
import numpy as np
import xgboost as xgb

class Predictor:
    def __init__(self):
        self.feature_cols = [
            'Home', 'Days_Rest', 'Return_From_Injury', 
            'Opp_Def_Rating', 'Opp_Pace', 'Blowout_Risk',
            'L3_PTS', 'L5_PTS', 'L10_PTS', 
            'L5_MIN', 'L5_FGA', 'L5_USG',
            'Season_Avg_PTS',
            'L5_PPM',
            'Trend_Multiplier'
        ]

    def convert_minutes(self, x):
        if pd.isna(x) or not isinstance(x, str): return 0.0
        try:
            if ":" in x:
                ind = x.index(":")
                return float(x[0:ind]) + (float(x[ind+1:ind+3]) / 60)
            return float(x)
        except:
            return 0.0

    def prepare_data(self, player_df, adv_stats, team_map, current_team_id):
        df = player_df.copy(deep=True)
        df['MP_Float'] = df['MP'].apply(self.convert_minutes)
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        
        df = df.assign(PPM=df['PTS'] / (df['MP_Float'] + 0.1))
        
        df = df.assign(Days_Rest=df['GAME_DATE'].diff().dt.days.fillna(3))
        
        recovery_values = []
        current_recovery = 0.0
        for rest in df['Days_Rest']:
            if rest >= 7:
                current_recovery = 1.0 
            elif current_recovery > 0:
                current_recovery = max(0.0, current_recovery - 0.2)
            recovery_values.append(current_recovery)
            
        df = df.assign(Return_From_Injury=recovery_values)
        df['Days_Rest'] = df['Days_Rest'].clip(upper=7)

        df['Opp_ID'] = df['Opp'].map(team_map)
        df = df.merge(adv_stats, left_on='Opp_ID', right_on='TEAM_ID', how='left')
        
        df = df.sort_values('GAME_DATE').reset_index(drop=True)
        df = df.rename(columns={'DEF_RATING': 'Opp_Def_Rating', 'PACE': 'Opp_Pace', 'NET_RATING': 'Opp_Net_Rating'})
        
        team_net_val = adv_stats.loc[adv_stats['TEAM_ID'] == current_team_id, 'NET_RATING']
        team_net_val = team_net_val.iloc[0] if not team_net_val.empty else 0.0
        
        df = df.assign(Blowout_Risk=(team_net_val - df['Opp_Net_Rating'].fillna(0)).abs())

        df['Opp_Def_Rating'] = df['Opp_Def_Rating'].fillna(115.0)
        df['Opp_Pace'] = df['Opp_Pace'].fillna(100.0)

        df = df.assign(
            L5_PPM=df['PPM'].rolling(window=5, min_periods=1).mean().shift(1),
            L3_PTS=df['PTS'].rolling(window=3, min_periods=1).mean().shift(1),
            L5_PTS=df['PTS'].rolling(window=5, min_periods=1).mean().shift(1),
            L10_PTS=df['PTS'].rolling(window=10, min_periods=1).mean().shift(1),
            L5_MIN=df['MP_Float'].rolling(window=5, min_periods=1).mean().shift(1),
            L5_FGA=df['FGA'].rolling(window=5, min_periods=1).mean().shift(1),
            L5_USG=(df['FGA'].rolling(5).sum().shift(1)) / (df['MP_Float'].rolling(5).sum().shift(1) + 0.1),
            Season_Avg_PTS=df['PTS'].expanding().mean().shift(1)
        )
        
        df = df.assign(Trend_Multiplier=df['L5_PTS'] / (df['Season_Avg_PTS'] + 0.1))

        df = df.bfill()
        df = df.fillna(0)

        return df

    def predict_next_game(self, player_df, adv_stats, team_map, current_team_id, next_game_data, experience):
        target_opp_id = next_game_data.get('Opp_ID')
        
        engineered_df = self.prepare_data(player_df, adv_stats, team_map, current_team_id)
        
        X = engineered_df[self.feature_cols].to_numpy()
        y = engineered_df['PTS'].to_numpy()
        
        n_samples = len(engineered_df)
        if experience == 0:
            weights = np.linspace(0.1, 4.0, n_samples) 
        elif experience <= 2:
            weights = np.linspace(0.5, 2.0, n_samples)
        else:
            weights = np.linspace(0.9, 1.1, n_samples)
            
        next_date = pd.to_datetime(next_game_data['Date'])
        last_game_date = pd.to_datetime(player_df.iloc[-1]['GAME_DATE'])
        next_game_rest = (next_date - last_game_date).days
        
        last_recovery_val = engineered_df.iloc[-1]['Return_From_Injury']
        next_recovery_val = 1.0 if next_game_rest >= 7 else max(0.0, last_recovery_val - 0.2)
        
        target_stats = adv_stats[adv_stats['TEAM_ID'] == target_opp_id]
        target_def_val = target_stats['DEF_RATING'].iloc[0] if not target_stats.empty else 115.0
        target_pace_val = target_stats['PACE'].iloc[0] if not target_stats.empty else 100.0
        target_net_val = target_stats['NET_RATING'].iloc[0] if not target_stats.empty else 0.0
        
        team_net_val = adv_stats.loc[adv_stats['TEAM_ID'] == current_team_id, 'NET_RATING']
        team_net_val = team_net_val.iloc[0] if not team_net_val.empty else 0.0
        next_blowout_risk = abs(team_net_val - target_net_val)
        
        current_l3_pts = player_df['PTS'].tail(3).mean()
        current_l5_pts = player_df['PTS'].tail(5).mean()
        current_l10_pts = player_df['PTS'].tail(10).mean()
        current_l5_min = player_df['MP'].apply(self.convert_minutes).tail(5).mean()
        current_l5_fga = player_df['FGA'].tail(5).mean()
        
        min_sum = player_df['MP'].apply(self.convert_minutes).tail(5).sum()
        current_l5_usg = player_df['FGA'].tail(5).sum() / (min_sum + 0.1)
        current_season_avg = player_df['PTS'].mean()
        
        current_ppm = player_df['PTS'].tail(5).sum() / (min_sum + 0.1)
        current_trend = current_l5_pts / (current_season_avg + 0.1)

        safe_l5 = 0.0 if pd.isna(current_l5_pts) else float(current_l5_pts)
        safe_avg = 0.0 if pd.isna(current_season_avg) else float(current_season_avg)
        
        dynamic_base = safe_l5 if experience == 0 else safe_avg
        if pd.isna(dynamic_base) or dynamic_base <= 0:
            dynamic_base = 0.5

        next_game_features = np.array([[
            next_game_data['Home'],                          
            min(next_game_rest, 7),                          
            next_recovery_val,                               
            target_def_val,                                  
            target_pace_val,         
            next_blowout_risk,       
            current_l3_pts,                                  
            current_l5_pts,                                  
            current_l10_pts,                                 
            current_l5_min,                                  
            current_l5_fga,                                  
            current_l5_usg,                                  
            current_season_avg,
            current_ppm,
            current_trend
        ]])

        xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror', 
            random_state=42,
            n_estimators=100,      
            learning_rate=0.05,    
            max_depth=3,
            subsample=0.8,
            base_score=dynamic_base,
            n_jobs=-1              
        )
        
        xgb_model.fit(X, y, sample_weight=weights)
        prediction = xgb_model.predict(next_game_features)[0]
        
        if experience == 0:
            prediction = (prediction * 0.6) + (safe_l5 * 0.4)
            
        return prediction
import pandas as pd
import time
from nba_api.stats.endpoints import synergyplaytypes

class PlayTypeEngine:
    _off_cache = None
    _def_cache = None

    def __init__(self):
        pass
        
    def pre_fetch_matrix(self, season_str='2025-26'):
        """Downloads the entire league matrix once to prevent slow per-player lookups."""
        if PlayTypeEngine._off_cache is not None:
            return
            
        print("Initializing Global Synergy Matrix...")
        off_dfs = []
        def_dfs = []
        play_types = ['Isolation', 'PRBallHandler', 'Spotup', 'Postup', 'Cut', 'Transition', 'PRRollMan', 'Handoff']
        
        for pt in play_types:
            try:
                time.sleep(0.6) 
                off = synergyplaytypes.SynergyPlayTypes(
                    play_type_nullable=pt, type_grouping_nullable='offensive', 
                    player_or_team_abbreviation='P', season=season_str
                ).get_data_frames()[0]
                off['PLAY_TYPE'] = pt
                off_dfs.append(off)
                
                time.sleep(0.6)
                dfn = synergyplaytypes.SynergyPlayTypes(
                    play_type_nullable=pt, type_grouping_nullable='defensive', 
                    player_or_team_abbreviation='T', season=season_str
                ).get_data_frames()[0]
                dfn['PLAY_TYPE'] = pt
                def_dfs.append(dfn)
                print(f" Cached {pt} data")
            except Exception as e:
                print(f" Could not cache {pt}: {e}")
                
        if off_dfs: PlayTypeEngine._off_cache = pd.concat(off_dfs, ignore_index=True)
        if def_dfs: PlayTypeEngine._def_cache = pd.concat(def_dfs, ignore_index=True)

    def calculate_matchup_delta(self, player_id, target_opp_id):
        """Ultra-fast lookup using the pre-fetched global cache."""
        if PlayTypeEngine._off_cache is None or PlayTypeEngine._def_cache is None:
            return 0.0
            
        player_off = PlayTypeEngine._off_cache[PlayTypeEngine._off_cache['PLAYER_ID'] == player_id]
        opp_def = PlayTypeEngine._def_cache[PlayTypeEngine._def_cache['TEAM_ID'] == target_opp_id]
        
        if player_off.empty or opp_def.empty: 
            return 0.0
            
        total_delta = 0.0
        for pt in ['Isolation', 'PRBallHandler', 'Spotup', 'Postup', 'Cut', 'Transition', 'PRRollMan', 'Handoff']:
            p_stats = player_off[player_off['PLAY_TYPE'] == pt]
            t_stats = opp_def[opp_def['PLAY_TYPE'] == pt]
            
            if not p_stats.empty and not t_stats.empty:
                try:
                    poss = float(p_stats.iloc[0]['POSS'])
                    opp_ppp = float(t_stats.iloc[0]['PPP'])
                    avg_ppp = PlayTypeEngine._def_cache[PlayTypeEngine._def_cache['PLAY_TYPE'] == pt]['PPP'].astype(float).mean()
                    total_delta += poss * (opp_ppp - avg_ppp)
                except: continue
                
        return max(-5.0, min(total_delta, 5.0))
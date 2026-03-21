import json
import os
from flask import Flask, render_template, request, jsonify, Response
from scraper import BasketballReferenceScraper
from predictor import Predictor
import pandas as pd

app = Flask(__name__)
scraper = BasketballReferenceScraper()
proc = Predictor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/teams', methods=['GET'])
def get_teams():
    try:
        nba_teams = scraper.get_all_teams()
        sorted_teams = sorted(nba_teams, key=lambda x: x['full_name'])
        return jsonify({"success": True, "teams": sorted_teams})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/stream_predict/<team_abbr>', methods=['GET'])
def stream_predict(team_abbr):
    def generate():
        try:
            # --- NEW: Load the Vegas Props Cache ---
            vegas_lines = {}
            if os.path.exists('vegas_props.json'):
                try:
                    with open('vegas_props.json', 'r') as f:
                        cache = json.load(f)
                        vegas_lines = cache.get('lines', {})
                except:
                    pass

            yield f"data: {json.dumps({'status': 'info', 'message': 'Fetching opponent schedule...'})}\n\n"
            next_game = scraper.scrape_next_game(team_abbr)
            if not next_game:
                yield f"data: {json.dumps({'status': 'error', 'message': 'No upcoming games found for this team.'})}\n\n"
                return
            
            nba_teams = scraper.get_all_teams()
            team_map = {t['abbreviation']: t['id'] for t in nba_teams}
            team_map['BRK'] = team_map.get('BKN')
            team_map['CHO'] = team_map.get('CHA')
            team_map['PHO'] = team_map.get('PHX')
            current_team_id = scraper.get_team_id(team_abbr)
                
            yield f"data: {json.dumps({'status': 'info', 'message': 'Fetching L20 Stats & DvP...'})}\n\n"
            adv_stats = scraper.scrape_advanced_team_stats()
            dvp_ranks = scraper.get_dvp_matrix() 
            
            opp_id = next_game.get('Opp_ID')
            team_stats = adv_stats[adv_stats['TEAM_ID'] == current_team_id] if 'TEAM_ID' in adv_stats.columns else pd.DataFrame()
            opp_stats = adv_stats[adv_stats['TEAM_ID'] == opp_id] if 'TEAM_ID' in adv_stats.columns else pd.DataFrame()
            
            team_net = team_stats['NET_RATING'].iloc[0] if not team_stats.empty and 'NET_RATING' in team_stats.columns else 0.0
            opp_net = opp_stats['NET_RATING'].iloc[0] if not opp_stats.empty and 'NET_RATING' in opp_stats.columns else 0.0
            blowout_risk = abs(team_net - opp_net)
            
            yield f"data: {json.dumps({'status': 'matchup_info', 'blowout_risk': round(blowout_risk, 1), 'team_net': round(team_net, 1), 'opp_net': round(opp_net, 1)})}\n\n"
            
            yield f"data: {json.dumps({'status': 'info', 'message': 'Pre-fetching metadata...'})}\n\n"
            team_meta_map = scraper.get_player_metadata(team_abbr)

            yield f"data: {json.dumps({'status': 'info', 'message': 'Mining active roster...'})}\n\n"
            projected_data = scraper.get_projected_lineup(team_abbr)
            
            if isinstance(projected_data, tuple) and len(projected_data) == 3:
                active_rotation, injured_out, projected_starters = projected_data
            else:
                active_rotation = projected_data[0] if isinstance(projected_data, tuple) else projected_data
                injured_out = []
                projected_starters = active_rotation[:5]
                
            if not active_rotation:
                yield f"data: {json.dumps({'status': 'warning', 'player': 'System Data', 'message': 'Failed to find active players.'})}\n\n"
                yield f"data: {json.dumps({'status': 'complete', 'message': 'Process stopped.'})}\n\n"
                return

            yield f"data: {json.dumps({'status': 'info', 'message': f'Executing Mega-Query for {len(active_rotation)} players...'})}\n\n"
            all_player_data = scraper.get_bulk_player_gamelogs(active_rotation)

            final_player_list = list(all_player_data.keys())
            total_players = len(final_player_list)
            
            for i, player in enumerate(final_player_list):
                yield f"data: {json.dumps({'status': 'progress', 'current': i, 'total': total_players, 'message': f'Training AI Ensemble for {player}...'})}\n\n"
                
                try:
                    player_data = all_player_data[player]
                    
                    metadata = team_meta_map.get(player, {'exp': 5, 'pos': 'F'})
                    experience = metadata['exp']
                    position = metadata['pos']
                    is_starter = player in projected_starters
                    
                    model_output = proc.predict_next_game(player_data, adv_stats, team_map, current_team_id, next_game, experience, position, dvp_ranks, is_starter)
                    
                    recent_games = player_data.tail(10)
                    history = []
                    for _, row in recent_games.iterrows():
                        raw_date = str(row['GAME_DATE']).split(' ')[0]
                        split_date = raw_date.split('-')
                        if len(split_date) >= 3:
                            clean_date = f"{split_date[1]}/{split_date[2]}"
                        else:
                            clean_date = raw_date
                        history.append({
                            "game_date": clean_date, 
                            "pts": int(row['PTS']),
                            "opp": str(row['Opp'])
                        })
                    
                    avg_10 = sum(h['pts'] for h in history) / len(history) if history else 0
                    
                    # --- FETCH VEGAS LINE FROM CACHE ---
                    # --- FETCH VEGAS LINE FROM CACHE ---
                    v_line = vegas_lines.get(player, None)
                    if v_line is None:
                        import unicodedata
                        def _normalize(name):
                            name = ''.join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')
                            name = name.lower().replace('.', '').replace('-', ' ').replace("'", "")
                            return name.replace(' jr', '').replace(' sr', '').replace(' iii', '').replace(' ii', '').strip()
                            
                        clean_player = _normalize(player)
                        p_parts = clean_player.split()
                        
                        for v_name, line in vegas_lines.items():
                            clean_v_name = _normalize(v_name)
                            v_parts = clean_v_name.split()
                            
                            if clean_player == clean_v_name:
                                v_line = line
                                break
                                
                            if len(p_parts) >= 2 and len(v_parts) >= 2:
                                if p_parts[-1] == v_parts[-1]: 
                                    f1, f2 = p_parts[0], v_parts[0]
                                    if f1 == f2 or (len(f1) >= 3 and f1 in f2) or (len(f2) >= 3 and f2 in f1):
                                        v_line = line
                                        break

                    result = {
                        "name": player,
                        "prediction": round(model_output["prediction"], 1),
                        "floor": round(model_output["floor"], 1),
                        "ceiling": round(model_output["ceiling"], 1),
                        "avg_10": round(avg_10, 1),
                        "vegas_line": v_line,  # <-- Passed to frontend
                        "history": history
                    }
                    
                    yield f"data: {json.dumps({'status': 'player_done', 'player_data': result, 'opponent': next_game['Opp']})}\n\n"
                    
                except Exception as e:
                    yield f"data: {json.dumps({'status': 'warning', 'player': player, 'message': str(e)})}\n\n"
                    continue
            
            yield f"data: {json.dumps({'status': 'complete', 'message': 'All predictions finalized!'})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"

    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
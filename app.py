import json
import os
import sys
import subprocess
import unicodedata
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response
import pandas as pd

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

from core.scraper import BasketballReferenceScraper
from core.predictor import Predictor

app = Flask(__name__)
scraper = BasketballReferenceScraper()
proc = Predictor()

def _normalize(name):
    name = ''.join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')
    name = name.lower().replace('.', '').replace('-', ' ').replace("'", "")
    return name.replace(' jr', '').replace(' sr', '').replace(' iii', '').replace(' ii', '').strip()

@app.route('/')
def index():
    return render_template('index.html')

# ==========================================
# Timer, Bets, Evaluators
# ==========================================

@app.route('/api/timer_status')
def timer_status():
    status_file = os.path.join(BASE_DIR, 'timer_status.json')
    if os.path.exists(status_file):
        try:
            with open(status_file, 'r') as f:
                return jsonify(json.load(f))
        except: pass
    return jsonify({"state": "offline"})

@app.route('/api/best_bets')
def best_bets():
    tracker_file = os.path.join(BASE_DIR, 'bet_tracker.csv')
    today_str = datetime.now().strftime('%Y-%m-%d')
    
    if not os.path.exists(tracker_file):
        return jsonify([])
        
    try:
        df = pd.read_csv(tracker_file)
        if 'Result' not in df.columns: return jsonify([])
        
        pending = df[(df['Result'] == 'PENDING') & (df['Date'] == today_str)].copy()
        if pending.empty: return jsonify([])

        player_team_map = {}
        slate_dir = os.path.join(BASE_DIR, 'Mega_Slate_Predictions')
        if os.path.exists(slate_dir):
            for file in os.listdir(slate_dir):
                if file.endswith('.csv'):
                    try:
                        m_df = pd.read_csv(os.path.join(slate_dir, file))
                        if 'Player' in m_df.columns and 'Team' in m_df.columns:
                            for _, r in m_df.iterrows():
                                player_team_map[r['Player']] = r['Team']
                    except: pass

        game_map = {}
        try:
            from nba_api.stats.endpoints import scoreboardv2
            board = scoreboardv2.ScoreboardV2().get_data_frames()[0]
            team_dict = {t['id']: t['abbreviation'] for t in scraper.get_all_teams()}
            
            for _, game in board.iterrows():
                home = team_dict.get(game['HOME_TEAM_ID'])
                away = team_dict.get(game['VISITOR_TEAM_ID'])
                status = str(game['GAME_STATUS_TEXT']).strip()
                
                sort_val = "23:59"
                display_time = status
                if "Final" not in status and "Half" not in status and "Q" not in status:
                    try:
                        clean_time = status.replace(' ET', '').strip()
                        parsed_time = datetime.strptime(clean_time, '%I:%M %p').time()
                        sort_val = parsed_time.strftime('%H:%M')
                        display_time = parsed_time.strftime('%I:%M %p')
                    except: pass
                
                g_info = {"title": f"{away} @ {home}", "time": display_time, "sort": sort_val}
                if home: game_map[home] = g_info
                if away: game_map[away] = g_info
        except:
            pass

        enriched_bets = []
        for _, row in pending.iterrows():
            bet = row.to_dict()
            team = player_team_map.get(bet['Player'], 'UNK')
            g_info = game_map.get(team, {"title": f"Game Matchup ({team})", "time": "TBD", "sort": "23:59"})
            
            bet['Team'] = team
            bet['GameTitle'] = g_info['title']
            bet['GameTime'] = g_info['time']
            bet['SortTime'] = g_info['sort']
            enriched_bets.append(bet)

        enriched_bets.sort(key=lambda x: (x['SortTime'], -x['Stars'], -abs(x['Edge'])))

        grouped = {}
        for bet in enriched_bets:
            g_key = f"{bet['GameTitle']} - {bet['GameTime']}"
            if g_key not in grouped: grouped[g_key] = []
            grouped[g_key].append(bet)

        return jsonify([{"game": k, "bets": v} for k, v in grouped.items()])
        
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/run_evaluator', methods=['POST'])
def run_evaluator():
    data = request.json
    script_name = data.get('script')
    args = data.get('args', [])
    
    script_path = os.path.join(BASE_DIR, 'evaluators', script_name)
    if not os.path.exists(script_path):
        return jsonify({"output": f"Error: Script {script_name} not found."})
        
    try:
        result = subprocess.run([sys.executable, script_path] + args, capture_output=True, text=True)
        output = result.stdout if result.stdout else result.stderr
        return jsonify({"output": output})
    except Exception as e:
        return jsonify({"output": f"Execution Error: {str(e)}"})

# ==========================================
# Web UI Predictions
# ==========================================

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
            today_str = datetime.now().strftime('%Y-%m-%d')
            
            vegas_lines = {}
            vegas_file = os.path.join(BASE_DIR, 'vegas_props.json')
            if os.path.exists(vegas_file):
                try:
                    with open(vegas_file, 'r') as f:
                        cache = json.load(f)
                        vegas_lines = cache.get('lines', {})
                except: pass

            yield f"data: {json.dumps({'status': 'info', 'message': 'Checking Database for cached predictions...'})}\n\n"
            mega_preds = {}
            master_file = os.path.join(BASE_DIR, 'Mega_Slate_Predictions', f'Master_Slate_{today_str}.csv')
            if os.path.exists(master_file):
                try:
                    m_df = pd.read_csv(master_file)
                    team_m_df = m_df[m_df['Team'] == team_abbr]
                    for _, r in team_m_df.iterrows():
                        pred_col = 'Predicted_PTS' if 'Predicted_PTS' in r else 'Pred_PTS' if 'Pred_PTS' in r else None
                        if pred_col: 
                            mega_preds[r['Player']] = {
                                'prediction': r[pred_col],
                                'floor': r.get('Floor', r.get('Floor_PTS', 0.0)),
                                'ceiling': r.get('Ceiling', r.get('Ceil_PTS', 0.0))
                            }
                except: pass

            yield f"data: {json.dumps({'status': 'info', 'message': 'Fetching opponent schedule...'})}\n\n"
            next_game = scraper.scrape_next_game(team_abbr)
            if not next_game:
                yield f"data: {json.dumps({'status': 'error', 'message': 'No upcoming games found.'})}\n\n"
                return
                
            target_opp = next_game.get('Opp', 'UNK')
            br_to_nba = {'BRK': 'BKN', 'CHO': 'CHA', 'PHO': 'PHX'}
            target_opp_nba = br_to_nba.get(target_opp, target_opp)

            nba_teams = scraper.get_all_teams()
            team_map = {t['abbreviation']: t['id'] for t in nba_teams}
            team_map.update(br_to_nba) 
            current_team_id = scraper.get_team_id(team_abbr)

            yield f"data: {json.dumps({'status': 'info', 'message': 'Fetching L20 Stats & DvP...'})}\n\n"
            adv_stats = scraper.scrape_advanced_team_stats()
            dvp_ranks = scraper.get_dvp_matrix()

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

            yield f"data: {json.dumps({'status': 'info', 'message': f'Executing Query for {len(active_rotation) + len(injured_out)} players...'})}\n\n"
            all_player_data = scraper.get_bulk_player_gamelogs(active_rotation + injured_out)

            total_vacated_pts = 0.0
            for inj_player in injured_out:
                if inj_player in all_player_data and not all_player_data[inj_player].empty:
                    season_avg = all_player_data[inj_player]['PTS'].mean()
                    if pd.notna(season_avg): total_vacated_pts += season_avg

            final_player_list = [p for p in active_rotation if p in all_player_data]
            total_players = len(final_player_list)

            for i, player in enumerate(final_player_list):
                try:
                    player_data = all_player_data[player].copy()
                    meta = team_meta_map.get(player, {'exp': 5, 'pos': 'F'})
                    is_starter = player in projected_starters

                    if player in mega_preds:
                        yield f"data: {json.dumps({'status': 'progress', 'current': i, 'total': total_players, 'message': f'Loading {player} from Master Slate...'})}\n\n"
                        prediction_val = float(mega_preds[player]['prediction'])
                        floor_val = float(mega_preds[player]['floor'])
                        ceil_val = float(mega_preds[player]['ceiling'])
                    else:
                        yield f"data: {json.dumps({'status': 'progress', 'current': i, 'total': total_players, 'message': f'Training AI for {player}...'})}\n\n"
                        model_output = proc.predict_next_game(
                            player_data, adv_stats, team_map, current_team_id, next_game,
                            meta['exp'], meta['pos'], dvp_ranks, is_starter, vacated_pts=total_vacated_pts
                        )
                        prediction_val = float(model_output["prediction"])
                        floor_val = float(model_output["floor"])
                        ceil_val = float(model_output["ceiling"])
                    
                    recent_10 = player_data['PTS'].tail(10).mean()
                    if pd.isna(recent_10): recent_10 = 0.0
                    
                    date_col = next((c for c in player_data.columns if 'DATE' in c.upper()), None)
                    if date_col:
                        player_data['TEMP_DATE'] = pd.to_datetime(player_data[date_col])
                        player_data = player_data.sort_values('TEMP_DATE').reset_index(drop=True)
                    else:
                        player_data = player_data.iloc[::-1].reset_index(drop=True)

                    matchup_col = next((c for c in player_data.columns if c.upper() in ['MATCHUP', 'OPP', 'OPPONENT']), None)
                    if matchup_col:
                        player_data['Opponent_Abbr'] = player_data[matchup_col].apply(lambda x: str(x).split(' ')[-1])
                    else:
                        player_data['Opponent_Abbr'] = 'UNK'
                        
                    if date_col:
                        player_data['Game_Date_Str'] = player_data['TEMP_DATE'].dt.strftime('%m/%d/%y')
                    else:
                        player_data['Game_Date_Str'] = 'UNK'

                    recent_10_pts = [float(x) for x in player_data['PTS'].tail(10).tolist()] if not player_data.empty else []
                    recent_10_opps = player_data['Opponent_Abbr'].tail(10).tolist() if not player_data.empty else []

                    h2h_df = player_data[player_data['Opponent_Abbr'] == target_opp_nba]
                    h2h_pts = [float(x) for x in h2h_df['PTS'].tail(5).tolist()] if not h2h_df.empty else []
                    h2h_dates = h2h_df['Game_Date_Str'].tail(5).tolist() if not h2h_df.empty else []

                    v_data = vegas_lines.get(player)
                    if v_data is None:
                        clean_player = _normalize(player)
                        for v_name, data in vegas_lines.items():
                            if _normalize(v_name) == clean_player:
                                v_data = data; break
                    
                    v_line = v_data.get('PTS') if isinstance(v_data, dict) else v_data

                    result = {
                        "name": player,
                        "prediction": round(prediction_val, 1),
                        "floor": round(floor_val, 1),
                        "ceiling": round(ceil_val, 1),
                        "avg_10": round(recent_10, 1),
                        "vegas_line": v_line,
                        "target_opp": target_opp,
                        "recent_10_pts": recent_10_pts,
                        "recent_10_opps": recent_10_opps,
                        "h2h_pts": h2h_pts,
                        "h2h_dates": h2h_dates
                    }

                    yield f"data: {json.dumps({'status': 'player_done', 'player_data': result})}\n\n"

                except Exception as e:
                    yield f"data: {json.dumps({'status': 'warning', 'player': player, 'message': str(e)})}\n\n"
                    continue

            yield f"data: {json.dumps({'status': 'complete', 'message': 'All predictions finalized!'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"

    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/save_csv', methods=['POST'])
def save_csv():
    try:
        data = request.json
        team_name = data.get('team', 'Unknown_Team')
        predictions = data.get('predictions', [])

        if not predictions: return jsonify({"success": False, "error": "No predictions to save."})

        now = datetime.now()
        folder_date = f"{now.month}-{now.day}-{now.year}"
        file_date = now.strftime('%Y-%m-%d')

        save_dir = os.path.join(BASE_DIR, "Testing_Predictions", folder_date)
        os.makedirs(save_dir, exist_ok=True)

        filepath = os.path.join(save_dir, f"{team_name}_Predictions_{file_date}.csv")
        
        exclude_keys = ['recent_10_pts', 'recent_10_opps', 'h2h_pts', 'h2h_dates', 'target_opp']
        clean_preds = [{k: v for k, v in p.items() if k not in exclude_keys} for p in predictions]
        pd.DataFrame(clean_preds).to_csv(filepath, index=False)

        return jsonify({"success": True, "path": filepath})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
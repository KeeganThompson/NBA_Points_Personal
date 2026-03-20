import json
from flask import Flask, render_template, request, jsonify, Response
from scraper import BasketballReferenceScraper
from predictor import Predictor

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
                
            yield f"data: {json.dumps({'status': 'info', 'message': 'Fetching advanced stats (Pace & Net Rating)...'})}\n\n"
            adv_stats = scraper.scrape_advanced_team_stats()
            
            yield f"data: {json.dumps({'status': 'info', 'message': 'Pre-fetching rookie data...'})}\n\n"
            team_exp_map = scraper.get_team_experience_data(team_abbr)

            yield f"data: {json.dumps({'status': 'info', 'message': 'Mining expected starting lineup...'})}\n\n"
            projected_data = scraper.get_projected_lineup(team_abbr)
            
            if isinstance(projected_data, tuple):
                active_rotation, injured_out = projected_data
            else:
                active_rotation = projected_data
                injured_out = []
                
            if not active_rotation:
                yield f"data: {json.dumps({'status': 'warning', 'player': 'System Data', 'message': 'Failed to find expected lineup.'})}\n\n"
                yield f"data: {json.dumps({'status': 'complete', 'message': 'Process stopped.'})}\n\n"
                return

            yield f"data: {json.dumps({'status': 'info', 'message': f'Executing Mega-Query for {len(active_rotation)} active players...'})}\n\n"
            all_player_data = scraper.get_bulk_player_gamelogs(active_rotation)

            final_player_list = list(all_player_data.keys())
            total_players = len(final_player_list)
            
            for i, player in enumerate(final_player_list):
                yield f"data: {json.dumps({'status': 'progress', 'current': i, 'total': total_players, 'message': f'Training XGBoost for {player}...'})}\n\n"
                
                try:
                    player_data = all_player_data[player]
                    experience = team_exp_map.get(player, 5)
                    
                    xgb_pts = proc.predict_next_game(player_data, adv_stats, team_map, current_team_id, next_game, experience)
                    
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
                    
                    result = {
                        "name": player,
                        "prediction": round(float(xgb_pts), 1),
                        "avg_10": round(avg_10, 1),
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
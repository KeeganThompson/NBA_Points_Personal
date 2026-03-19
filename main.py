import time
from scraper import BasketballReferenceScraper
from predictor import Predictor

if __name__ == "__main__":
    scraper = BasketballReferenceScraper()
    proc = Predictor()

    player_name = input("Enter NBA Player Name (e.g. 'Luka Doncic'): ")
    
    try:
        player_data = scraper.scrape_player_gamelog(player_name)
        
        current_team_abbr = player_data.iloc[-1]['Tm']
        print(f"{player_name} plays for {current_team_abbr}.")
        time.sleep(1) 
        
        team_data = scraper.scrape_team_gamelog(current_team_abbr)
        time.sleep(1)

        print("Fetching current opponent defensive ratings...")
        def_ratings = scraper.scrape_league_defensive_ratings()
        time.sleep(1)
        
        next_game = scraper.scrape_next_game(current_team_abbr)
        
        if next_game:
            print(f"\nNext Game Found: vs {next_game['Opp']} (Home: {bool(next_game['Home'])}) on {next_game['Date']} at Hour: {next_game['Time']}\n")
            
            xgb_pts, best_params = proc.predict_next_game(player_data, team_data, def_ratings, next_game)
            
            print("- - - - - - - - - ADVANCED NEXT GAME PREDICTIONS - - - - - - - - -")
            print(f"XGBoost Prediction: {xgb_pts:.2f} PTS")
            print(f"\n[Optimum Hyperparameters Used: {best_params}]")
            
    except Exception as e:
        print(f"An error occurred: {e}")
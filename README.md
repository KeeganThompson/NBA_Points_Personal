# NBA Player Points Generator

An automated, machine learning-driven architecture designed to predict NBA player performance, identify inefficiencies in live Vegas betting markets, and backtest quantitative betting strategies. 

This system uses a dual-model ensemble (**XGBoost** and **LightGBM**) with dynamic hyperparameter tuning via **Optuna** to generate median projections, absolute floors, and ceilings for every active NBA player. It features an automated data pipeline that scrapes the NBA API, ESPN injury reports, and The Odds API to build a highly contextual 25+ variable feature set (including pace-adjustments, vacated usage, and Synergy play-type mismatch data).

---

## Directory Structure

* `Mega_Slate_Predictions/`: The historical archive of all automated 200+ player daily slate projections.
* `Sportsbooks_Lines/`: The historical "Time Machine" archive of DraftKings closing lines.
* `Testing_Predictions/`: Saved outputs generated manually via the Web UI.

---

## The Core Engine

### `scraper.py`
**What it does:** The primary data ingestion pipeline. It interacts directly with the NBA API to pull historical box scores, advanced defensive ratings, and DvP (Defense vs. Position) matrices. It aggressively scrapes ESPN for live injury reports (treating Day-to-Day as 'Out' to dynamically recalculate vacated usage) and Rotowire for projected starting lineups. 
**How to run:** Inherited automatically by other scripts. Not run standalone.

### `predictor.py`
**What it does:** The machine learning brain. It engineers the raw data into 25+ advanced features (EWMA, Schedule Fatigue, Pace-Adjusted Per-100s). It uses Optuna to run real-time simulations to adapt learning rates and tree depths to individual player variance histories before passing the data to the XGBoost and LightGBM models.
**How to run:** Inherited automatically by other scripts.

### `app.py`
**What it does:** The Flask backend for the interactive Web UI. It streams the Optuna training process to the frontend in real-time, allowing for manual testing and visual analysis of specific teams.
**How to run:** * `python app.py` (Then open `localhost:5000` in your browser).

---

## The Automated Snipers

### `predict_timer.py`
**What it does:** The fully automated "Set and Forget" orchestrator. When launched, it scans today's NBA schedule, groups games by tip-off time, and sleeps in the background. Exactly 30 minutes before a game begins (when injury reports are finalized), it wakes up and automatically triggers `odds_fetcher.py` and `mega_slate.py` for those specific teams.
**How to run:** * `python predict_timer.py`

### `odds_fetcher.py`
**What it does:** Connects to The Odds API to download live consensus player prop lines. It saves a temporary cache for the Web UI and permanently archives a timestamped copy into the `Sportsbooks_Lines` folder for historical grading.
**How to run:** * `python odds_fetcher.py` (Fetches the entire league).
* `python odds_fetcher.py --teams MIA,LAL` (Precision snipe for specific teams to save API credits).

### `mega_slate.py`
**What it does:** The mass-generator. It scans the active rosters, runs the AI ensemble for every playing individual, compares the predictions to the Vegas lines, and logs any 3-Star+ edges to `bet_tracker.csv`. It saves the raw output to the `Mega_Slate_Predictions` folder.
**How to run:** * `python mega_slate.py` (Runs all teams playing today).
* `python mega_slate.py --teams MIA,LAL` (Runs targeted teams and *appends* them to today's master file).

### `slate_merger.py`
**What it does:** A data stitcher. If you manually test teams via the Web UI, this script gathers those isolated CSVs from the `Testing_Predictions` folder and seamlessly merges/overwrites them into the official daily `Master_Slate` without deleting the rest of the league.
**How to run:**
* `python slate_merger.py`

---

## Strategy & Evaluators

### `bet_analyzer.py`
**What it does:** The strategic testing ground and daily ledger. It grades your live bets against actual box scores and allows you to test different mathematical betting strategies against your historical archive.
**How to run:**
* `python bet_analyzer.py --grade` (Checks NBA box scores and updates pending bets in `bet_tracker.csv`).
* `python bet_analyzer.py --gradeall` (Sweeps historical archives using the Standard Strategy: Tailing all 3-Star, 4-Star, and 5-Star AI edges).
* `python bet_analyzer.py --optimized` (Sweeps historical archives using the Hybrid Strategy: Fading the AI on marginal 3/4-Star leans, but Tailing the AI on absolute 5-Star locks).

### `mega_evaluator.py`
**What it does:** Measures the raw accuracy (MAE - Mean Absolute Error) of the AI Ensemble directly against Vegas closing lines to see who predicted the actual box score better.
**How to run:**
* `python mega_evaluator.py --file "Mega_Slate_Predictions/Master_Slate_2026-03-22.csv"` (Evaluates a single specific day).
* `python mega_evaluator.py --gradeall` (Mass evaluation of all historical slates against all historical DraftKings lines).

### `fade_evaluator.py`
**What it does:** The Contrarian Grader. It evaluates a hypothetical betting system where you exclusively bet *against* the AI model whenever it disagrees with Vegas by 2.5+ points, assuming the sportsbook is setting a trap.
**How to run:**
* `python fade_evaluator.py --file <path>` (Evaluates a single day).
* `python fade_evaluator.py --gradeall` (Mass historical fade evaluation).
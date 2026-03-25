import os
import sys
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

def merge_daily_predictions():
    now = datetime.now()
    folder_date = f"{now.month}-{now.day}-{now.year}"
    file_date = now.strftime('%Y-%m-%d')

    input_dir = os.path.join(BASE_DIR, "Testing_Predictions", folder_date)
    output_dir = os.path.join(BASE_DIR, "Mega_Slate_Predictions")
    output_file = os.path.join(output_dir, f"Master_Slate_{file_date}.csv")

    print("=======================================================")
    print("   SLATE MERGER: COMBINING WEB UI & MEGA SLATE")
    print("=======================================================")

    df_list = []

    # Check if Mega-Slate already exists
    if os.path.exists(output_file):
        print(f" 📄 Found existing Master Slate from predict_timer: {output_file}")
        try:
            existing_master = pd.read_csv(output_file)
            df_list.append(existing_master)
        except Exception as e:
            print(f"    Error reading existing Master Slate: {e}")
    else:
        print("  No existing Master Slate found. Building from scratch...")

    # Check for Web preds
    if os.path.exists(input_dir):
        csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
        if csv_files:
            print(f"  Found {len(csv_files)} Web UI team files. Merging...")
            for file in csv_files:
                file_path = os.path.join(input_dir, file)
                try:
                    df = pd.read_csv(file_path)
                    
                    if 'name' in df.columns:
                        df = df.rename(columns={
                            'name': 'Player',
                            'prediction': 'Predicted_PTS',
                            'floor': 'Floor',
                            'ceiling': 'Ceiling',
                            'avg_10': '10_Game_Avg',
                            'vegas_line': 'Vegas_Line'
                        })
                    
                    df_list.append(df)
                    print(f"    Added {file}")
                except Exception as e:
                    print(f"    Error reading {file}: {e}")
        else:
            print(f"  No Web UI CSV files found in {input_dir}.")
    else:
        print(f"  Web UI folder {input_dir} does not exist today.")

    # Combine and Save
    if not df_list:
        print("  No valid data could be found to merge.")
        return

    master_df = pd.concat(df_list, ignore_index=True)
    
    # Drop dupes.
    # overwrite morning Mega-Slate runs if same player exists in both
    initial_count = len(master_df)
    master_df = master_df.drop_duplicates(subset=['Player'], keep='last')
    dupes_removed = initial_count - len(master_df)

    os.makedirs(output_dir, exist_ok=True)
    master_df.to_csv(output_file, index=False)

    print("-------------------------------------------------------")
    print(f"  Successfully stitched {len(master_df)} unique players into the Master Slate.")
    if dupes_removed > 0:
        print(f"  Updated/Overwrote {dupes_removed} entries with fresh Web UI data.")
    print(f"  Saved to: {output_file}")
    print("=======================================================\n")

if __name__ == "__main__":
    merge_daily_predictions()
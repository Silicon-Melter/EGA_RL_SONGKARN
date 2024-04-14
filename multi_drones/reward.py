
import numpy as np
from utils.calculation_utils import yaw_diff_nomalized
from utils.airsim_plotting import draw_text
from utils.airsim_utils import to_vec3r
import math
from rich import print
import pandas as pd
import os
import datetime
now = datetime.datetime.now()
formatted_datetime = now.strftime("%d.%m.%y-%H%M")

def computeReward(client, episodeLog, distance_before, distance_now, goal_rad, cur_pry, bef_pry, cur_pos, client_id):
    r = -2.0
    distance_before = episodeLog['distance_from_goal'][-1]

    before_track_diff = abs(yaw_diff_nomalized(bef_pry[2], goal_rad)) # track_diff [0, pi]
    after_track_diff = abs(yaw_diff_nomalized(cur_pry[2], goal_rad)) # track_diff [0, pi]

    distance_diff = distance_before - distance_now # if closer + , further -

    yaw_rew = yaw_reward(before_track_diff,after_track_diff)
    distance_rew = 5 * distance_diff

    # dis_rew = distance_reward(distance_diff, 10, distance_now)

    r +=  yaw_rew
    r +=  distance_rew

    if abs(distance_now - distance_before) < 0.001:
        r = r - 1.0
        print("not moving  -1  ")
    
    if abs(distance_now - distance_before) < 0.001:
        r = r - 1.0
        print("not moving  -1  ")
    folder_path = r"C:\Users\noppa\Documents\GitHub\AI_logs\rewLogs\\" + formatted_datetime + "\\"

    # Create the folder if it doesn't exist
    try:
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    except FileExistsError:
        print(f"Folder '{folder_path}' already exists.")

    # Define the data dictionary
    data = {
        "distance_diff": [distance_diff],
        "before_track_diff": [before_track_diff],
        "after_track_diff": [after_track_diff],
        "distance_reward": [distance_rew],
        "yaw_reward": [yaw_rew],
        "reward": [r]
    }

    # Construct the file path
    log_dir = folder_path + "data" + str(client_id) + ".xlsx"

    # Check if the file exists
    if not os.path.exists(log_dir):
        # Create a new DataFrame with the data
        df = pd.DataFrame(data)
        # Write the DataFrame to an Excel file
        df.to_excel(log_dir, index=False)
        print(f"Excel file '{log_dir}' created successfully.")
    else:
        # Read the existing Excel file into a DataFrame
        existing_df = pd.read_excel(log_dir)
        print(f"Excel file '{log_dir}' already exists.")
        # Create a new DataFrame with the data
        new_df = pd.DataFrame(data)
        # Concatenate the existing DataFrame and the new DataFrame
        df = pd.concat([existing_df, new_df], ignore_index=True)

        # Write the concatenated DataFrame back to the Excel file
        df.to_excel(log_dir, index=False)
        print(f"Data appended to '{log_dir}' successfully.")
    return r 



def yaw_reward(before_track_diff, after_track_diff):
    #sub-reward from getting closer to 0 degree
    before = -5*before_track_diff**2+10
    after = -5*after_track_diff**2+10
    #see for improvement
    return after+0.5*(after-before)


import numpy as np
from utils.calculation_utils import yaw_diff_nomalized
from utils.airsim_plotting import draw_text
from utils.airsim_utils import to_vec3r
import math
import pandas as pd

# reward as a result of taking actions
def computeReward(client, distance_before, distance_now, goal_rad, cur_pry, bef_pry, cur_pos):
    #r = -2.0 # for doing nothing
    r = 0
    
    before_track_diff = abs(yaw_diff_nomalized(bef_pry[2], goal_rad)) # track_diff [0, pi]
    after_track_diff = abs(yaw_diff_nomalized(cur_pry[2], goal_rad)) # track_diff [0, pi]

    distance_diff = distance_before - distance_now # if closer + , further -

    yaw_rew = yaw_reward(before_track_diff,after_track_diff)
    distance_rew = 5 * distance_diff

    # dis_rew = distance_reward(distance_diff, 10, distance_now)

    r +=  yaw_rew
    r +=  distance_rew
    
    
    draw_text(
        client,
        [
            f"deg_diff: {math.degrees(after_track_diff)}", 
            f"yaw_rew:  {yaw_rew}",
            # f"dis_diff: {distance_diff}",
            # f"dis_rew:  {dis_rew}"
        ],
        [
            to_vec3r(cur_pos), 
            to_vec3r((cur_pos[0], cur_pos[1], cur_pos[2]+0.2)),
            # to_vec3r((cur_pos[0], cur_pos[1], cur_pos[2]+0.4)),
            # to_vec3r((cur_pos[0], cur_pos[1], cur_pos[2]+0.6)),
        ]
        
    )
   

    if abs(distance_now - distance_before) < 0.001:
        r = r - 1.0
        print("not moving  -1  ")
    df = pd.read_csv(r"C:\Users\EGA\Documents\GitHub\AI_logs\rewLogs\data.xlsx", encoding="utf-8")  # Try 'utf-8' encoding
    data = {
        "distance_diff": [distance_diff],
        "before_track_diff": [before_track_diff],
        "after_track_diff": [after_track_diff],
        "distance_reward": [distance_rew],
        "yaw_reward": [yaw_rew],
        "reward": [r]
    }
    print(data)
    df = df._append(pd.DataFrame(data), ignore_index=True)
    df.to_excel(r"C:\Users\EGA\Documents\GitHub\AI_logs\rewLogs\data.xlsx", index=False)
    return r 


def yaw_reward(before_track_diff, after_track_diff):
    #sub-reward from getting closer to 0 degree
    before = -1*before_track_diff**2+10
    after = -1*after_track_diff**2+10
    #see for improvement
    return after+0.5*(after-before)

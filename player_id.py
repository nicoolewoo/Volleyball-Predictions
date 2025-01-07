from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd 
#ok i might have done this entire function for no reason at all
THRESHOLD = 50 #determines whether detected player in frame is considered same as another
players = {}
next_id = 0

def assign_players(frame, prev_frame, img_id):
    global players, next_id
    # i want a new set of players for each video
    if 'prev_img_id' not in assign_players.__dict__:
        assign_players.prev_img_id = None
    if frame.iloc[0]["Image Filename"] != assign_players.prev_img_id:
        next_id = 0
        players = {}
    #find bounding box centers
    assign_players.prev_img_id = img_id
    curr_center = np.array([(row['X'] + row['Width'] / 2, row['Y'] + row['Height'] / 2)
                                for _, row in frame.iterrows()])
    prev_center = np.array([(row['X'] + row['Width'] / 2, row['Y'] + row['Height'] / 2)
                                 for _, row in prev_frame.iterrows()]) if not prev_frame.empty else np.empty((0, 2))
    #match players 
    if prev_center.shape[0] > 0: #if no players detected move on
        distances = cdist(curr_center, prev_center, metric='euclidean') #returns matrix of distances 
        # print(f"Distances between current and previous frame:\n{distances}")
        matched_ind = distances < THRESHOLD #find potential matches 
        # print(f"Matched indices (within threshold):\n{matched_ind}")
        frame['Player ID'] = -1 #default for new
        for i, curr_player in frame.iterrows():
            if i < matched_ind.shape[0]:
                matched_indices = np.where(matched_ind[i])[0]  # Find indices of matched players
                if matched_indices.size > 0:
                # Assign ID of the closest matched player
                    matched_idx = matched_indices[0]  # Choose the closest match
                    matched_id = prev_frame.iloc[matched_idx]['Player ID']
                    frame.at[i, 'Player ID'] = matched_id

                    players[matched_id] = {
                        'center': curr_center[i],
                        'last_seen_frame': frame.iloc[0]['Image Filename']
                    }
                else:
                #ID for unmatched players
                    frame.at[i, 'Player ID'] = next_id
                    players[next_id] = {
                        'center': curr_center[i],
                        'last_seen_frame': frame.iloc[0]['Image Filename']
                    }
                    next_id += 1
            
    else: # we are on the first frame
        frame['Player ID'] = range(next_id, next_id + len(frame))
        for i, row in frame.iterrows():
            players[next_id] = {
                'center': ([row["X"] + (row["Width"] / 2), row["Y"] + (row["Height"] / 2)]),
                'last_seen_frame': row['Image Filename']
            }
            next_id += 1
    return frame



import pandas as pd 
import os
from pathlib import Path
#each video is a folder (0-55), within each folder is a frame ID XXXXXX, and within each frame ID is
#41 images, 

def dataset_ball(url):
    data = []
    url = Path(url)
    train_videos = {1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54}
    val_videos = {0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51}
    test_videos = {4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47}

    for video_id in range(55):
        video_dir = url / str(video_id)
        if not video_dir.exists():
            print(f"Directory does not exist: {video_dir}")
            continue
        txt_files = sorted(video_dir.glob("*.txt"), key=lambda x: int(x.stem))
        # print(f"Processing files in video {video_id}: {txt_files}")

        if not txt_files:
            continue
        for frame_dir in txt_files:
            try:
                frame_id = int(frame_dir.stem)
            except ValueError:
                continue
            frame_path = str(frame_dir)
            with open(frame_path, "r") as images:
                coordinates = images.readlines() #by row 
                if not coordinates:
                    print(f"No coordinates found in {frame_path}")
                    continue
                for frame_ind, pixels in enumerate(coordinates): 
                    try:
                        x, y = map(int, pixels.strip().split()) #by white space
                    except ValueError:
                        print(f"Invalid pixel data: '{pixels}' in file {frame_path}")
                        continue
                    ball_in_frame = not (x == 0 and y == 0)
                    data.append({
                        "video_id": video_id,
                        "frame_id": frame_id,
                        "frame_time": frame_ind,
                        "x_coord": x,
                        "y_coord": y,
                        "ball_visible": ball_in_frame
                    })

    df = pd.DataFrame(data)
    return df

def dataset_detections(url): 
    root_dir = url
    data =[]
    for video_dir in root_dir.iterdir():
        if video_dir.is_dir():
            frame_dirs = sorted(video_dir.iterdir(), key=lambda x:int(x.name) if x.is_dir() else float('inf'))
            for frame_dir in frame_dirs:
                if frame_dir.is_dir():
                    action_file = frame_dir / "action_detections.txt"
                    if action_file.exists():
                        with action_file.open("r") as f:
                            row_counter = 0
                            for line in f:
                                fields = line.strip().split("\t")
                                image_filename = fields[0]
                                action_class_id = int(fields[1])                                
                                for i in range(2, len(fields), 6):  #Each bounding box has 6 fields
                                    try:
                                        x = int(fields[i])
                                        y = int(fields[i + 1])
                                        width = int(fields[i + 2])
                                        height = int(fields[i + 3])
                                        confidence = float(fields[i + 4])
                                        action_label = fields[i + 5]
                                        
                                        data.append([
                                            video_dir.name, frame_dir.name, image_filename, action_class_id,
                                            x, y, width, height, confidence, action_label, row_counter
                                        ])
                                        
                                        # print(data)
                                    except (IndexError, ValueError) as e:
                                        print(f"Error processing line: {line.strip()}")
                                        print(f"Exception: {e}")
                                        continue
                                row_counter += 1 #please work im about to have a stroke bruh
                            
        columns = [
            "video_id", "frame_id", "Image Filename", "Action Class ID",
            "X", "Y", "Width", "Height", "Confidence", "Action Label", "frame_time"
        ]
        df = pd.DataFrame(data, columns=columns)
        return df






            




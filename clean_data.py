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
        txt_files = list(video_dir.glob("*.txt"))
        if not txt_files:
            continue
        for frame_dir in txt_files:
            try:
                frame_id = int(frame_dir.stem)
                # print(f"Processing frame_id: {frame_id}")
            except ValueError:
                # print("Can't find frame name: ", frame_dir)
                continue
            frame_path = str(frame_dir)
            with open(frame_path, "r") as images:
                coordinates = images.readline().strip()
                if not coordinates:
                    print(f"No coordinates found in {frame_path}")
                    continue
                for frame_ind, pixels in enumerate(coordinates.split(';')): 
                    try:
                        x, y = map(int, pixels.strip().split())
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
    data = []
    root = url
    for video_dir in root.iterdir():
        for frames in video_dir.iterdir():
            if frames.is_dir():
                action_file = frames / "action_detections.txt"
                if action_file.exists():
                    with action_file.open("r") as f:
                        for line in f:
                            fields = line.strip().split("\t")
                            img = fields[0]
                            action_class_id = int(fields[1])
                            for i in range (2, len(fields), 7):
                                x = int(fields[i])
                                y = int(fields[i + 1])
                                width = int(fields[i + 1])
                                height = int(fields[i + 3])
                                confidence = float(fields[i + 4])
                                action_label = fields[i + 5]
                                data.append({
                                    "Video ID": video_dir.name,
                                    "Frame ID": frames.name,
                                    "Image Filename": img,
                                    "Action Class ID": action_class_id,
                                    "X": x,
                                    "Y": y,
                                    "Width": width,
                                    "Height": height,
                                    "Confidence": confidence,
                                    "Action Label": action_label
                                })
    df = pd.DataFrame(data)
    return df





            




import os
import json
import cv2
import numpy as np
import torch
from tqdm import tqdm
import shutil
import argparse

def clear_directory(directory_path):
    """Clear all files and subdirectories in the specified directory. Create the directory if it does not exist."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    else:
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove file or symbolic link
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove directory
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

def extract_data_from_json(json_path):
    """Extract action data from JSON for visualizing keys."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data["actions"]

def process_videos_and_metadata(video_dir, metadata_dir, output_metadata_dir, threshold=0.1, height_threshold=0.1):
    # Clear output directories at the start
    clear_directory(output_metadata_dir)

    for video_file in tqdm(os.listdir(video_dir)):
        if not video_file.endswith('.mp4'):
            continue
        
        video_name = os.path.splitext(video_file)[0]
        json_file = os.path.join(metadata_dir, f"{video_name}.json")
        output_json_file = os.path.join(output_metadata_dir, f"{video_name}.json")

        if not os.path.exists(json_file):
            print(f"Metadata file for {video_name} not found. Skipping.")
            continue
        
        with open(json_file, 'r') as f:
            metadata = json.load(f)
        
        video_path = os.path.join(video_dir, video_file)
        actions = metadata.get('actions', {})
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_file}")
            continue
        
        ret, prev_frame = cap.read()
        if not ret:
            print(f"Failed to read frames from video: {video_file}")
            cap.release()
            continue

        # Initialize default fields for all actions
        for frame_idx in range(len(actions)):
            actions[str(frame_idx)]['collision'] = 0  # Initialize single collision flag
            actions[str(frame_idx)]['jump_invalid'] = 0
            actions[str(frame_idx)]['delta_pos'] = [0.0, 0.0, 0.0]

        # First pass: mark jumps and collisions
        for frame_idx in range(1, len(actions)):  # Exclude the first frame
            current_action = actions[str(frame_idx)]
            prev_action = actions.get(str(frame_idx - 1), None)

            if prev_action:
                # Calculate delta pos
                delta_pos = np.array(current_action['pos']) - np.array(prev_action['pos'])
                current_action['delta_pos'] = delta_pos.tolist()

                # Mark jump as invalid if height change is too small
                if current_action.get('scs') == 1 and delta_pos[1] <= height_threshold:
                    current_action['jump_invalid'] = 1

                # Mark collision if pos[0] and pos[2] changes are both below the threshold
                if abs(delta_pos[0]) <= threshold and abs(delta_pos[2]) <= threshold:
                    current_action['collision'] = 1

        # Second pass: mark subsequent jumps in a sequence as invalid
        jump_sequence_started = False
        for frame_idx in range(1, len(actions)):  # Exclude the first frame
            current_action = actions[str(frame_idx)]

            if current_action.get('scs') == 1 and current_action['jump_invalid'] == 0:
                if jump_sequence_started:  # If already in a sequence, mark as invalid
                    current_action['jump_invalid'] = 1
                else:  # First valid jump in a sequence
                    jump_sequence_started = True
            else:
                jump_sequence_started = False  # Reset sequence if no jump or invalid jump

        metadata['actions'] = actions
        with open(output_json_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        cap.release()

def main():
    # Set up argparse to handle command-line arguments
    parser = argparse.ArgumentParser(description="Process videos and metadata.")
    parser.add_argument('--dir_name', type=str, help="Root directory for the video and metadata files.")
    parser.add_argument('--threshold', type=float, default=0.01, help="Threshold for detecting collisions.")
    parser.add_argument('--height_threshold', type=float, default=0.01, help="Threshold for jump validity based on height change.")

    args = parser.parse_args()

    # Get root_name from command line argument
    dir_name = args.dir_name
    threshold = args.threshold
    height_threshold = args.height_threshold

    video_dir = os.path.join(dir_name, "video")
    metadata_dir = os.path.join(dir_name, "metadata")
    output_metadata_dir = os.path.join(dir_name, "metadata-detection")

    process_videos_and_metadata(video_dir, metadata_dir, output_metadata_dir, threshold, height_threshold)


if __name__ == "__main__":
    main()
import cv2
import os
def read_video(video_path):
    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        return []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video: {video_path}")
        return []
    frames = []
    while True:
        flag, frame = cap.read()
        if not flag:
            break
        frames.append(frame)
    cap.release()
    print(f"[INFO] Total frames read: {len(frames)}")
    return frames

def save_video(output_video_frames, output_video_path):

    if not output_video_frames:
        print("No frames to write! Skipping video save.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    height, width = output_video_frames[0].shape[:2]
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (width, height))
    for frame in output_video_frames:
        out.write(frame)
    out.release()
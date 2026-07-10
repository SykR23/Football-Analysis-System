import os.path
import numpy as np
import pandas as pd
import cv2
from ultralytics import YOLO
import supervision as sv
import pickle
from utils import get_width, get_center, get_foot_position

class Tracker: 
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def add_position_to_track(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position = get_center(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', {}) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {'bbox': x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detection_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detection_batch
        return detections

    def get_detection_track(self, frames, stub_path=None, read_from_stub=False):
        if stub_path and read_from_stub is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks = {
            'players': [],
            'referees': [],
            'ball': []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inverse = {v:k for k,v in cls_names.items()}
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # convert goalkeepers to players
            for object_ind, cls_id in enumerate(detection_supervision.class_id):
                if cls_names[cls_id] == 'goalkeeper':
                    detection_supervision.class_id[object_ind] = cls_names_inverse["player"]

            # Track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks['players'].append({})
            tracks['referees'].append({})
            tracks['ball'].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                if cls_id == cls_names_inverse['player']:
                    tracks['players'][frame_num][track_id] = {'bbox': bbox}
                if cls_id == cls_names_inverse['referee']:
                    tracks['referees'][frame_num][track_id] = {'bbox': bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                if cls_id == cls_names_inverse['ball']:
                    tracks['ball'][frame_num][1] = {'bbox': bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center(bbox)
        width = get_width(bbox)

        cv2.ellipse(frame, center=(x_center, y2), axes = (int(width), int(0.35*width)),angle=0.0, startAngle=-45, endAngle=255,
                    color=color, thickness=2, lineType=cv2.LINE_4)

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width//2
        y1_rect = (y2 - rectangle_height//2) + 10
        x2_rect = x_center + rectangle_width//2
        y2_rect = (y2 + rectangle_height//2) + 10

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)
            x1_text = x1_rect + 12
            if track_id>99:
                x1_text -= 10

            cv2.putText(
                frame,
                f'{track_id}',
                (int(x1_text), int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )
        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x,_ = get_center(bbox)
        triangle_points = np.array([
            [x, y],
            [x-10, y-20],
            [x+10, y-20]
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2)
        return frame

    def draw_rounded_rectangle(self, frame, top_left, bottom_right, color, radius=20, thickness=2, alpha=0.4):
        overlay = frame.copy()
        output = frame.copy()

        # Draw filled rounded rectangle on overlay
        x1, y1 = top_left
        x2, y2 = bottom_right

        # Use OpenCV's rectangle for a basic effect
        cv2.rectangle(overlay, top_left, bottom_right, color, -1, cv2.LINE_AA)

        # Blend overlay with original frame
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

        # Draw border
        cv2.rectangle(output, top_left, bottom_right, (50, 50, 50), thickness, cv2.LINE_AA)

        return output

    def draw_possession_bar(self, frame, x, y, width, height, team_1_ratio):
        team_1_color = (227.81568627, 231.50196078, 233.81176471)  # white
        team_2_color = (0, 255, 0)  # green

        team_1_width = int(width * team_1_ratio)
        team_2_width = width - team_1_width

        cv2.rectangle(frame, (x, y), (x + team_1_width, y + height), team_2_color, -1)
        cv2.rectangle(frame, (x + team_2_width, y), (x + width, y + height), team_1_color, -1)

        # Outline
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 0), 2)

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        team_ball_control_till_frame = team_ball_control[:frame_num+1]

        # get no of times each team has the ball
        team_1_num_frames = np.sum(team_ball_control_till_frame == 1)
        team_2_num_frames = np.sum(team_ball_control_till_frame == 2)

        total = team_1_num_frames + team_2_num_frames
        if total > 0:
            team_1 = team_1_num_frames / (team_1_num_frames + team_2_num_frames)
            team_2 = team_2_num_frames / (team_1_num_frames + team_2_num_frames)
        else:
            team_1 = team_2 = 0

        # Draw background box
        frame = self.draw_rounded_rectangle(frame, (1350, 850), (1900, 970), (255, 255, 255), radius=10, thickness=2,
                                           alpha=0.3)
        # Draw possession bar
        self.draw_possession_bar(frame, 1370, 970, 500, 20, team_1)

        cv2.putText(frame, f'Team 1 possession: {team_1*100:.2f}%', (1400, 900), cv2.FONT_HERSHEY_DUPLEX, 1,(0, 0, 0),3)
        cv2.putText(frame, f'Team 2 possession: {team_2*100:.2f}%', (1400, 950), cv2.FONT_HERSHEY_DUPLEX, 1,(0, 0, 0),3)

        return frame

    def draw_annotations(self, frames, tracks, team_ball_control):
        output_vid_frames = []
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            player_dict = tracks['players'][frame_num]
            referee_dict = tracks['referees'][frame_num]
            ball_dict = tracks['ball'][frame_num]

            # draw players
            for track_id, player in player_dict.items():
                color = player.get('team_color', (0,0,255))
                frame = self.draw_ellipse(frame, player['bbox'], color, track_id)

                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player['bbox'], (255, 0, 0))

            # draw referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee['bbox'], (0, 255, 255))
            # draw ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball['bbox'], (0, 255, 0))
            # draw team ball control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_vid_frames.append(frame)
        return output_vid_frames
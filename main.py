import numpy as np
import cv2
from utils import read_video
from Tracker import Tracker
from team_assigner import Assigner
from assign_player_ball import PlayerBallAssigner
from Camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransform
from Speed_distance_estimator import SpeedDistEstimator


def main():
    video_frames = read_video('Input_Videos/08fd33_4.mp4')

    # initialise tracker
    tracker = Tracker('model/best(1).pt')
    tracks = tracker.get_detection_track(video_frames, read_from_stub=True, stub_path='Stubs/track_stubs.pkl')

    # get object positions
    tracker.add_position_to_track(tracks)

    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                              read_from_stub=True,
                                                                              stub_path='Stubs/camera_movement.pkl')
    camera_movement_estimator.adjust_position_to_tracks(tracks, camera_movement_per_frame)

    # view transformer
    view_transformer = ViewTransform()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # interpolate ball positions
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

    # assign player teams
    team_assigner = Assigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # assign player aquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            if team_ball_control:
                team_ball_control.append(team_ball_control[-1])
            else:
                team_ball_control.append(-1)
    team_ball_control = np.array(team_ball_control)

    # estimate speed and distance
    estimator = SpeedDistEstimator()
    estimator.add_speed_distance_to_Tracks(tracks)

    # draw object tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    # draw speed and distance
    estimator.draw_speed_distance(output_video_frames, tracks)

    # set up video writer
    height, width, _ = output_video_frames[0].shape
    video_writer = cv2.VideoWriter('Output_videos/output_vid.avi',
                                   cv2.VideoWriter_fourcc(*'XVID'),
                                   30, (width, height))

    # draw camera movement on updated frames
    camera_movement_estimator.draw_camera_movement(output_video_frames,
                                                   camera_movement_per_frame,
                                                   video_writer)

    video_writer.release()

    # save_video(output_video_frames, 'Output_videos/output_vid.avi')


if __name__ == '__main__':
    main()

from utils import get_center, get_distance

class PlayerBallAssigner():
    def __init__(self):
        self.max_player_distance = 60

    def assign_ball_to_player(self, players, ball_bbox):
        ball_position = get_center(ball_bbox)

        minimum_distance = 99999
        assigned_player = -1

        for player_id, player in players.items():
            player_bbox = player['bbox']

            distance_left = get_distance((player_bbox[0], player_bbox[-1]), ball_position)
            distance_right = get_distance((player_bbox[2], player_bbox[-1]), ball_position)
            distance = min(distance_right, distance_left)

            if distance < self.max_player_distance:
                if distance < minimum_distance:
                    minimum_distance = distance
                    assigned_player = player_id

        return assigned_player
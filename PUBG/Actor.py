import numpy as np
from itertools import compress

COLLECT_FROM_ADV_DIST = 0.08

class PlayerActor(object):
    def __init__(self, field_width, num_advantage, player_name):
        self.player_name = player_name
        self.pos = np.random.uniform(-field_width/2, field_width/2, size=2)
        self.speed = 0.05
        self.power = 100
        self.kill = 0
        self.heading_adv = None
        self.notmet_advantage_point = [True for _ in range(num_advantage)]

    def move(self, r, advantage_points):
        norm = np.linalg.norm(self.pos)
        if norm > r:
            dp = - self.pos / norm
        else:
            advantage_points_filter = np.array(list(compress(advantage_points, self.notmet_advantage_point)))
            dist_advantange = np.linalg.norm(advantage_points_filter, axis=1)
            advantage_points_filter = advantage_points_filter[np.argwhere(dist_advantange < r).flatten()]
            if len(advantage_points_filter) == 0:
                advantage_points_filter = advantage_points[-1:]
            vector_to_advantage = advantage_points_filter - self.pos
            dist_to_advantange = np.linalg.norm(vector_to_advantage, axis=1)
            idx = np.argmin(dist_to_advantange)
            # print(vector_to_advantage.shape, dist_to_advantange.shape)
            dp = vector_to_advantage[idx] / dist_to_advantange[idx]

        self.pos += dp * self.speed

    def check_advantage_point(self, advantage_points, advantage_points_met):
        vector_to_advantage = advantage_points - self.pos
        dist_to_advantange = np.linalg.norm(vector_to_advantage, axis=1)
        idx = np.argmin(dist_to_advantange)
        if dist_to_advantange[idx] < COLLECT_FROM_ADV_DIST:
            self.notmet_advantage_point[idx] = False
            if not advantage_points_met[idx]:
                self.power += 30
                advantage_points_met[idx] = True

            
    

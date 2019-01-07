import numpy as np
from gym.envs.classic_control import rendering

action_set = [[0, 1], [1, 0], [0, -1], [-1, 0]]


class Pose():
    def __init__(self, x, y):
        self.x = x
        self.y = y

        self.x0 = x
        self.y0 = x


    def get_pose_xy(self):
        return [self.x, self.y]


    def chg_pose(self, xy):
        self.x = xy[0]
        self.y = xy[1]


    def update_posexy(self, dx, dy):
        self.x = self.x + dx
        self.y = self.y + dy


    def set_intial_pose(self):
        self.x0 = self.x
        self.y0 = self.y

class OccupancyMap:
    def __init__(self, map_size, lspan, prior, chance):
        self.omap = prior*np.ones(map_size)
        self.omap = np.pad(self.omap, pad_width=5, mode='constant', constant_values= 1)
        self.lspan = lspan
        self.chance = chance

    def get_local_omap(self, pose):
        tmp = self.omap[(pose[0]-self.lspan):(pose[0]+self.lspan+1), (pose[1]-self.lspan):(pose[1]+self.lspan+1)]
        return tmp

    def get_omap(self):
        return self.omap

    def update_omap(self, sensor_v):
        self.omap = self.omap + sensor_v
        self.omap = np.clip(self.omap,0,1)

    def update_omap_one_free(self, pose):
        self.omap[pose[0], pose[1]] = 0

    def update_omap_one_add(self,pose,value):
        pose = pose.astype(int)
        ff = np.round(self.omap[pose[0]][pose[1]] + value,1)
        self.omap[pose[0], pose[1]] = np.clip(ff,0,1)


class EnvironmentMap:
    def __init__(self, map_path='small_map.txt', mspan=1, lspan=2, episode_length=400):
        self.map_path = map_path
        self.pose = Pose(5, 5)                                      # Can be any defualt starting position
        self.cnt = None
        self.mspan = mspan
        self.lspan = lspan
        self.entropy_sum = None
        self.episode_length = episode_length

        self.chance = [1,1]

        self.viewer = None

    def reset(self):
        map_load = np.loadtxt(self.map_path , dtype=float, delimiter=',')
        self.omap = OccupancyMap(np.shape(map_load), self.lspan, 0.5, self.chance)

        self.map = np.pad(map_load, pad_width=5, mode='constant', constant_values=1)
        self.map_size = np.shape(self.map)

        self.cnt = 0

        #Generate random starting pose
        while True:
            x = np.random.randint(0, self.map_size[0])
            y = np.random.randint(0, self.map_size[1])
            o = np.random.randint(0, 4)
            if self.checkpose([x, y]):
                self.pose.chg_pose([x, y, o])
                self.pose.set_intial_pose()
                self.omap.update_omap_one_free(self.pose.get_pose_xy())
                break

        sensor = self.get_sensor()
        self.omap.update_omap(sensor)

        self.entropy_sum = np.sum(self.get_entropy(self.omap.get_omap()))

        return self.get_obs()


    def get_sensor(self):
        sensor = np.zeros(self.map_size)

        x_low, x_high = self.pose.get_pose_xy()[0] - self.mspan, self.pose.get_pose_xy()[0] + self.mspan
        y_low, y_high = self.pose.get_pose_xy()[1] - self.mspan, self.pose.get_pose_xy()[1] + self.mspan

        for i in range(x_low, x_high + 1):
            for j in range(y_low, y_high + 1):
                if self.map[i, j] == 0:
                    sensor[i, j] = -0.5
                else:
                    sensor[i, j] = 0.5

        return sensor

    def get_obs(self):
        local_map = self.omap.get_local_omap(self.pose.get_pose_xy())

        local_entropy_map = self.get_entropy(local_map)

        local_map = (local_map - 0.5) * 2

        local_entropy_map /= -np.log(.5)
        local_entropy_map = np.round((local_entropy_map - 0.5)*2,2)

        return np.concatenate((np.expand_dims(local_map, axis=0), np.expand_dims(local_entropy_map, axis=0)), axis=0)

    def log_to_prob(self, log_t):
        prob = 1 - 1/(1+np.exp(log_t))
        return prob

    def get_entropy(self, p_t):
        H = -p_t * np.log(p_t) - (1 - p_t) * np.log(1 - p_t)
        H = np.nan_to_num(H)
        return H

    def checkpose(self, xy):
        if self.map[xy[0], xy[1]] == 1:
            return False
        else:
            return True

    def step(self, action):
        if self.cnt is None:
            print("Must call env.reset() before calling step()")
            return
        self.cnt += 1

        done = False
        meta = False
        # Select action
        [dx, dy] = action_set[action]
        self.pose.update_posexy(dx, dy)
        # Check pose is valid
        if self.checkpose(self.pose.get_pose_xy()):
            # Recieve sensor value, n times
            sensor_t = self.get_sensor()
            self.omap.update_omap(sensor_t)
            # Caluclate new map entropy
            tmp = np.sum(self.get_entropy(self.omap.get_omap()))

            # Calcultate difference in entropy from old map and return reward
            #reward = np.exp(self.entropy_sum - map_entropy_sum)
            reward = (self.entropy_sum - tmp)
            self.entropy_sum = tmp
            if self.cnt == self.episode_length:
                done = True
        else:
            reward = -5
            meta = True
            done = True
        return self.get_obs(), reward, done, meta

    def render(self, reset=False):
        wide = self.map_size[0]
        height = self.map_size[1]

        if reset:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None

        if self.viewer is None:
            self.viewer = rendering.Viewer(1000, 500)
            self.viewer.set_bounds(0, 2 * wide, 0, height)

            self.geom_grid = []
            for i in range(wide):
                geoms = []
                for j in range(height):
                    geoms.append(self.make_box(i + wide, j, 1, 1))
                    self.viewer.add_geom(geoms[-1])
                    if self.map[i, j] == 1:
                        poly = self.make_box(i, j, 1, 1, color=(.5, .5, .5))
                        self.viewer.add_geom(poly)
                self.geom_grid.append(geoms)

            self.pos = self.make_box(self.pose.x, self.pose.y, 1, 1, color=(1., 105. / 255, 180. / 255))
            self.postrans = rendering.Transform()
            self.postrans.set_translation(self.pose.x0, self.pose.y0)
            self.pos.add_attr(self.postrans)
            self.viewer.add_geom(self.pos)

            self.pos1 = self.make_box(self.pose.x, self.pose.y, 1, 1, color=(1., 105. / 255, 180. / 255))
            self.postrans1 = rendering.Transform()
            self.postrans1.set_translation(self.pose.x0 + wide, self.pose.y0)
            self.pos1.add_attr(self.postrans1)
            self.viewer.add_geom(self.pos1)

        p = self.omap.get_omap()

        for i in range(wide):
            for j in range(height):
                self.geom_grid[i][j].set_color(0, p[i, j], 0)

        self.postrans.set_translation(self.pose.x - self.pose.x0, self.pose.y - self.pose.y0)
        self.postrans1.set_translation(self.pose.x - self.pose.x0 + wide, self.pose.y - self.pose.y0)

        return self.viewer.render(return_rgb_array=True)

    def make_box(self, x, y, w, h, color=None):
        from gym.envs.classic_control import rendering

        poly = rendering.make_polygon([(x, y), (x, y + h), (x + w, y + h), (x + w, y)], filled=True)
        if color is not None:
            poly.set_color(*color)

        return poly

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
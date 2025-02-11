import numpy as np
import cv2

class Environment:

    def __init__(self, dim):
        self.dim = dim
        self.scaler = dim/651
        self.env = np.array([[[0, 0, 0]] * dim] * dim)
        self.img = self.env.astype(np.uint8)
        self.BGR = (100, 100, 100)
        self.obstacle_position = None
        self.obstacle_dim = None

    def build_environment(self, num_blocks, num_walls):
        self.build_blocks(num_blocks)
        self.build_walls(num_walls)
        return self.img

    def show_environment(self):
        cv2.imshow('image', self.img)
        cv2.waitKey(0)

    def build_blocks(self, num_blocks):
        dims = np.random.randint(1, 100, (num_blocks, 2))
        for i in range(len(dims)):
            position = np.random.randint(1, self.dim-1, (1, 2))
            cv2.rectangle(self.img,
                          (position[0][0], position[0][1]),
                          (position[0][0] + dims[i][0], position[0][1] + dims[i][1]), self.BGR, -1)

    def build_walls(self, num_walls):
        dims = (np.array([[5, 100],[100, 5]])*self.scaler).astype(int)
        for i in range(num_walls):
            position = np.random.randint(1, self.dim-1, (1, 2))
            scale = np.random.randint(1, 3)
            cv2.rectangle(self.img,
                          (position[0][0], position[0][1]),
                          (position[0][0] + dims[i%2][0]*scale, position[0][1] + dims[i%2][1]*scale), self.BGR, -1)

    def build_default(self):
        dims = (np.array([[250, 20], [100, 20]])*self.scaler).astype(int)
        pts = (np.array([400, 150, 275, 125, 500, -5])*self.scaler).astype(int)
        # draw scaled points and dims
        cv2.rectangle(self.img, (pts[0], pts[1]), (pts[2]+dims[0][0], pts[1]+dims[0][1]), self.BGR, -1)
        cv2.rectangle(self.img, (pts[1], pts[0]), (pts[1]+dims[0][0], pts[0]+dims[0][1]), self.BGR, -1)
        cv2.rectangle(self.img, (pts[1], pts[0]), (pts[1]+dims[1][1], pts[0]+dims[1][0]), self.BGR, -1)
        cv2.rectangle(self.img, (pts[1]+dims[0][0], pts[0]-dims[1][0]), (pts[1]+dims[0][0]+dims[1][1], pts[0]+dims[0][1]+dims[1][0]), self.BGR, -1)
        cv2.rectangle(self.img, (pts[0], pts[1]), (pts[0]+dims[1][1], pts[1]+dims[1][0]-25), self.BGR, -1)
        cv2.rectangle(self.img, (pts[2], pts[3]), (pts[2]+dims[1][1], pts[3]+dims[1][0]*2), self.BGR, -1)
        cv2.rectangle(self.img, (pts[2], pts[4]), (pts[2]+dims[1][1], pts[4]+dims[1][0]*2), self.BGR, -1)
        cv2.rectangle(self.img, (pts[1], pts[5]), (pts[1]+dims[1][1], pts[5]+dims[1][0]*2), self.BGR, -1)
        return self.img

    def build_dynamic_obstacle(self):
        self.obstacle_position = (np.array([80, 200])*self.scaler).astype(int)
        self.obstacle_dim = (np.array([15])*self.scaler).astype(int)
        # draw obstacle
        cv2.circle(self.img, (self.obstacle_position[0], self.obstacle_position[1]), int(self.obstacle_dim[0]), (0,0,255), -1)
        return self.img

    def get_obstacle_position(self):
        return self.obstacle_position

    def get_obstacle_dim(self):
        return self.obstacle_dim

    def invert_environment(self, env):
        for i in range(len(env)):
           for j in range(len(env[i])):
                for k in range(len(env[i][j])):
                    env[i][j][k] = np.abs(env[i][j][k]-255)
        return env
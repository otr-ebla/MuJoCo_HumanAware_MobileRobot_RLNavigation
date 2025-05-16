import os
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco_viewer
from torch.utils.tensorboard import SummaryWriter

class mobilerobotRL(gym.Env):
    def __init__(self, num_rays = 108, training = True, log_dir="runs/default") -> None:
        super().__init__()
        self.training = training
        self.num_rays = num_rays
        self.max_episode_steps = 1000
        self.current_step = 0
        self.previous_distance = 100
        self.episode_return = 0
        self.mean_episode_return = 0
        self.episode_counter = 0
        self.success_counter = 0
        self.collision_counter = 0
        self.timeout_counter = 0
        self.last_episode_result = None
        self.episode_time_length = 0
        self.episode_time_begin = 0
        self.lidar_readings = None  
        self.success_rate = 0
        self.collision_rate = 0
        self.timeout_rate = 0
        self.robot_relative_azimuth = 0

        # Mobile Robot action space
        self.action_space = gym.spaces.Box(
            low = np.array([0, -1.0]),
            high = np.array([1.0, 1.0]),
            shape = (2, ),
            dtype = np.float32
        )

        # Mobile Robot observation space
        self.observation_space = gym.spaces.Box(
            low = np.array([0.0]*num_rays+[0.0, -np.pi]),
            high = np.array([30.0]*num_rays+[0.0, np.pi]),
            shape = (num_rays + 2, ),
            dtype = np.float32
        )

        self.writer = SummaryWriter(log_dir=log_dir)

    def _get_obs(self, robot_pos, target_pos, robot_rot_matrix, lidar_readings):
        self.lidar_readings = lidar_readings
        self.lidar_readings = np.array(lidar_readings)
        self.lidar_readings = lidar_readings.flatten()

        robot_forward_vector = robot_rot_matrix[:, 0]
        robot_yaw_angle = np.arctan2(robot_forward_vector[1], robot_forward_vector[0])

        relative_position = target_pos - robot_pos
        distance_target_robot = np.linalg.norm(relative_position[:2])
        global_robot_azimuth = np.arctan2(relative_position[1], relative_position[0])

        self.robot_relative_azimuth = global_robot_azimuth - robot_yaw_angle
        self.robot_relative_azimuth = (self.robot_relative_azimuth+np.pi)%(2*np.pi)-np.pi

        obs = np.concatenate((self.lidar_readings, [distance_target_robot, self.robot_relative_azimuth]))
        return obs.astype(np.float32)
    
    def update_episode_metrics(self, result: str):
        self.episode_counter += 1
        if result == "success":
            self.success_counter += 1
        elif result == "collision":
            self.collision_counter += 1
        elif result == "timeout":
            self.timeout_counter += 1

        if self.episode_counter > 0:
            self.success_rate = self.success_counter / self.episode_counter
            self.collision_rate = self.collision_counter / self.episode_counter
            self.timeout_rate = self.timeout_counter / self.episode_counter

        # Log metrics to TensorBoard
        self.writer.add_scalar("metrics/episode_return", self.episode_return, self.episode_counter)
        self.writer.add_scalar("metrics/sucess_rate", self.success_rate, self.episode_counter)
        self.writer.add_scalar("metrics/collision_rate", self.collision_rate, self.episode_counter)
        self.writer.add_scalar("metrics/timeout_rate", self.timeout_rate, self.episode_counter)

        # Reset the episode
        self.episode_return = 0
        self.mean_episode_return = 0
        self.last_episode_result = None
        self.episode_time_length = 0
        self.episode_time_begin = time.time()

    def log_step_reward(self, reward):
        self.writer.add_scalar("metrics/step_reward", reward, self.current_step)

    def reset(self, seed=None, options=None):
        raise NotImplementedError("Implementarion of reset method is required in the derived class.")
    
    def step(self, action):
        raise NotImplementedError("Implementarion of step method is required in the derived class.")
    
    def render(self, mode='human'):
        pass
    
    def check_collision(self):
        raise NotImplementedError("Implementarion of check_collision method is required in the derived class.")
    
    def close(self):
        self.writer.close()
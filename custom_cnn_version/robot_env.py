import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
from collections import deque
import cv2
import time

class RobotArmEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode='rgb_array', image_size=64, frame_skip=4, frame_stacks=4, max_steps=200):
        super().__init__()
        self.render_mode = render_mode
        self.image_size = image_size
        self.frame_skip = frame_skip
        self.frame_stacks = frame_stacks
        self.max_steps = max_steps

        # PyBullet
        self.physicsClient = p.connect(p.GUI if render_mode == 'human' else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self.physicsClient)

        # load plane and robot
        self.planeId = p.loadURDF("plane.urdf", physicsClientId=self.physicsClient)
        self.robotId = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True, physicsClientId=self.physicsClient)
        self.end_effector_index = 11

        # small cube target
        self.target_object_id = p.loadURDF("cube.urdf", globalScaling=0.08, physicsClientId=self.physicsClient)

        # initial joints
        self.initial_joint_positions = [0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0]
        for i in range(7):
            p.resetJointState(self.robotId, i, self.initial_joint_positions[i], physicsClientId=self.physicsClient)

        # camera
        self.viewMatrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.5, 0, 0.2],
            distance=0.7,
            yaw=90,
            pitch=-45,
            roll=0,
            upAxisIndex=2,
            physicsClientId=self.physicsClient
        )
        self.projMatrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=1.0,
            nearVal=0.1,
            farVal=10.0,
            physicsClientId=self.physicsClient
        )

        # observation_space: pixels channels_first (C, H, W) and proprioception (7,)
        pixel_shape = (self.frame_stacks, self.image_size, self.image_size)
        self.observation_space = spaces.Dict({
            "pixels": spaces.Box(low=0.0, high=1.0, shape=pixel_shape, dtype=np.float32),
            "proprioception": spaces.Box(low=-np.pi, high=np.pi, shape=(7,), dtype=np.float32)
        })

        # action: delta x,y,z in [-1,1] -> scaled inside env
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # internal
        self._frame_buffer = deque(maxlen=self.frame_stacks)
        self.current_step = 0
        self.last_distance = None
        self.next_seed = None

        # dynamics tuning
        self.max_delta = 0.03  # meters per action (scaled)
        self.max_distance_norm = 1.0  # used to normalize distance

    # ---------- Helpers ----------
    def _render_frame(self):
        img_arr = p.getCameraImage(
            width=self.image_size,
            height=self.image_size,
            viewMatrix=self.viewMatrix,
            projectionMatrix=self.projMatrix,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self.physicsClient
        )
        if img_arr is None or img_arr[2] is None:
            gray = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        else:
            rgb = np.reshape(img_arr[2], (self.image_size, self.image_size, 4))[:, :, :3]
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        # normalize to [0,1]
        frame = gray.astype(np.float32) / 255.0
        return frame  # shape (H, W)

    def _get_proprio(self):
        joint_states = p.getJointStates(self.robotId, list(range(7)), physicsClientId=self.physicsClient)
        joint_positions = np.array([s[0] for s in joint_states], dtype=np.float32)
        return joint_positions

    def _get_observation(self):
        # frame buffer -> channels_first
        stacked = np.stack(list(self._frame_buffer), axis=0)  # (C, H, W)
        proprio = self._get_proprio()
        return {"pixels": stacked.astype(np.float32), "proprioception": proprio.astype(np.float32)}

    def _apply_action(self, action):
        # action: [-1,1]^3 -> delta in meters
        delta = np.clip(action, -1.0, 1.0) * self.max_delta
        link_state = p.getLinkState(self.robotId, self.end_effector_index, physicsClientId=self.physicsClient)
        current_pos = np.array(link_state[0])
        target_pos = current_pos + delta

        # keep Z >= 0.0 (table) and limit workspace
        target_pos[2] = max(0.02, target_pos[2])
        target_pos[0] = float(np.clip(target_pos[0], 0.2, 0.8))
        target_pos[1] = float(np.clip(target_pos[1], -0.4, 0.4))

        orientation = p.getQuaternionFromEuler([0, -np.pi, 0])
        target_joints = p.calculateInverseKinematics(
            self.robotId,
            self.end_effector_index,
            target_pos.tolist(),
            orientation,
            maxNumIterations=100,
            physicsClientId=self.physicsClient
        )
        for i in range(7):
            p.setJointMotorControl2(
                bodyUniqueId=self.robotId,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_joints[i],
                force=500,
                physicsClientId=self.physicsClient
            )

    # ---------- Gym API ----------
    def step(self, action):
        # apply and simulate
        self._apply_action(np.asarray(action, dtype=np.float32))
        for _ in range(self.frame_skip):
            p.stepSimulation(physicsClientId=self.physicsClient)

        # update frames
        frame = self._render_frame()
        self._frame_buffer.append(frame)

        # increment
        self.current_step += 1

        # compute reward
        link_state = p.getLinkState(self.robotId, self.end_effector_index, physicsClientId=self.physicsClient)
        tool_pos = np.array(link_state[0])
        obj_pos, _ = p.getBasePositionAndOrientation(self.target_object_id, physicsClientId=self.physicsClient)
        obj_pos = np.array(obj_pos)
        distance = float(np.linalg.norm(tool_pos - obj_pos))
        # contact
        contact_points = p.getContactPoints(self.robotId, self.target_object_id, self.end_effector_index, physicsClientId=self.physicsClient)
        is_contact = len(contact_points) > 0

        # smooth normalized distance reward (closer -> higher)
        dist_norm = np.clip(distance / self.max_distance_norm, 0.0, 1.0)
        dist_reward = 1.0 - dist_norm  # in [0,1]

        # delta shaping
        if self.last_distance is None:
            delta = 0.0
        else:
            delta = self.last_distance - distance  # positive if got closer
        self.last_distance = distance

        # contact bonus moderate
        contact_reward = 5.0 if is_contact else 0.0

        # small step penalty to encourage efficiency
        step_penalty = -0.01

        reward = float(dist_reward + 2.0 * delta + contact_reward + step_penalty)
        # clip to avoid huge spikes
        reward = float(np.clip(reward, -10.0, 10.0))

        # termination
        terminated = bool(is_contact or self.current_step >= self.max_steps)
        truncated = False
        info = {"distance": distance, "is_contact": is_contact}

        obs = self._get_observation()
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # seed handling
        if hasattr(self, 'next_seed') and self.next_seed is not None:
            seed = self.next_seed
            self.next_seed = None

        # gymnasium seed
        super().reset(seed=seed)
        # use self.np_random for sampling
        # reset simulation
        p.resetSimulation(physicsClientId=self.physicsClient)
        p.setGravity(0, 0, -9.81, physicsClientId=self.physicsClient)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.planeId = p.loadURDF("plane.urdf", physicsClientId=self.physicsClient)
        self.robotId = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True, physicsClientId=self.physicsClient)
        self.target_object_id = p.loadURDF("cube.urdf", globalScaling=0.08, physicsClientId=self.physicsClient)

        # reset joints
        for i in range(7):
            p.resetJointState(self.robotId, i, self.initial_joint_positions[i], physicsClientId=self.physicsClient)

        # sample target position within workspace using controlled RNG
        rand_x = float(self.np_random.uniform(low=0.35, high=0.65))
        rand_y = float(self.np_random.uniform(low=-0.25, high=0.25))
        p.resetBasePositionAndOrientation(self.target_object_id, [rand_x, rand_y, 0.02], [0, 0, 0, 1], physicsClientId=self.physicsClient)

        # step some steps to stabilize
        for _ in range(50):
            p.stepSimulation(physicsClientId=self.physicsClient)

        # initialize frame buffer
        self._frame_buffer.clear()
        base_frame = self._render_frame()
        for _ in range(self.frame_stacks):
            self._frame_buffer.append(base_frame)

        self.current_step = 0
        self.last_distance = None

        obs = self._get_observation()
        info = {}
        return obs, info

    def render(self):
        # GUI handled by pybullet
        pass

    def set_seed(self, seed):
        # called from your evaluate / replay harness
        self.next_seed = seed

    def close(self):
        if p.isConnected(physicsClientId=self.physicsClient):
            p.disconnect(physicsClientId=self.physicsClient)
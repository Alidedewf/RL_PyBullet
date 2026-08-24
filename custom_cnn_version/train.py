import os
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from robot_env import RobotArmEnv
from custom_cnn import CustomCombinedExtractor

LOG_DIR = "./logs/robot_arm_visual_ppo"
os.makedirs(LOG_DIR, exist_ok=True)

if __name__ == "__main__":
    # create env factory so each vector env gets its own instance with consistent params
    def make_env_fn():
        return RobotArmEnv(render_mode='rgb_array', image_size=64, frame_skip=4, frame_stacks=4, max_steps=200)

    n_envs = 8
    vec_env = make_vec_env(lambda: make_env_fn(), n_envs=n_envs, vec_env_cls=DummyVecEnv)

    # policy kwargs: use our extractor
    FEATURES_DIM = 512
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(features_dim=FEATURES_DIM),
        net_arch=[dict(pi=[256, 128], vf=[256, 128])]
    )

    model = PPO(
        "MultiInputPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=2048,         # can tune
        batch_size=64,
        learning_rate=2.5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.005,
        vf_coef=0.5,
        tensorboard_log=LOG_DIR,
        device="cuda" if th.cuda.is_available() else "cpu"
    )

    # callbacks
    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=LOG_DIR, name_prefix="rl_model")
    eval_env = DummyVecEnv([lambda: RobotArmEnv(render_mode='rgb_array', image_size=64, frame_skip=4, frame_stacks=4)])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(LOG_DIR, "best_model"),
        log_path=LOG_DIR,
        eval_freq=10000 // n_envs,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    total_timesteps = 600_000
    try:
        model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, eval_callback], progress_bar=True)
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        model.save(os.path.join(LOG_DIR, "final_model.zip"))
        print("Model saved.")
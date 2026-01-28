import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
import os
import numpy as np
from vpp_env import UrbanVPPEnv

def linear_schedule(initial_value: float):
    """Linear learning rate schedule."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func 


def main():
    # === TRAINING CONFIGURATION ===
    # Customize episode timing: start_hour controls when each episode begins
    # None = random start (default), or specify hour (0-23) for consistent training
    start_hour = None  # e.g., 11 for 11am, None for random
    episode_length = 96  # Number of steps (96 = 24 hours)
    
    # Safety: Create directories
    os.makedirs("./checkpoints/", exist_ok=True)
    os.makedirs("./tensorboard_logs/", exist_ok=True)
    os.makedirs("./eval_logs/", exist_ok=True)
    
    print("[INFO] Checking environment validity...")
    check_env(UrbanVPPEnv(data_path="./data"), warn=True)
    
    # 1. Create Training Environments (Multiple for parallel training)
    n_envs = 4  # Use 4 parallel environments for faster training
    
    # Calculate start_step from start_hour if specified
    start_step = start_hour * 4 if start_hour is not None else None
    
    def make_env():
        def _init():
            env = UrbanVPPEnv(data_path="./data")
            env = Monitor(env)
            return env
        return _init
    
    # Custom wrapper class for timed episodes (defined at module level for pickling)
    class TimedResetWrapper(gym.Wrapper):
        def __init__(self, env, start_step_val, episode_len_val):
            super().__init__(env)
            self.start_step_val = start_step_val
            self.episode_len_val = episode_len_val
        
        def reset(self, **kwargs):
            if 'options' not in kwargs:
                kwargs['options'] = {}
            kwargs['options']['start_step'] = self.start_step_val
            kwargs['options']['episode_len'] = self.episode_len_val
            return self.env.reset(**kwargs)
    
    # Apply wrapper if start_step is specified
    if start_step is not None:
        def make_env_timed():
            def _init():
                env = UrbanVPPEnv(data_path="./data")
                env = TimedResetWrapper(env, start_step, episode_length)
                env = Monitor(env)
                return env
            return _init
        env_factory = make_env_timed
    else:
        env_factory = make_env
    
    # Use SubprocVecEnv for true parallel execution (faster on multi-core CPUs)
    env = SubprocVecEnv([env_factory() for _ in range(n_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=5.0, clip_reward=10.0)
    
    # 2. Create Evaluation Environment (Separate to track actual performance)
    eval_env = DummyVecEnv([env_factory()])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=5.0, training=False)
    
    print("[OK] Environment created and normalized.")
    print(f"[INFO] Training with {n_envs} parallel environments")

    # 2. Define the PPO Model with improved architecture
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256])  # Deeper network for complex VPP dynamics
    )
    
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=linear_schedule(3e-4),  # Decaying LR for better convergence
        gamma=0.99,                # Discount factor
        gae_lambda=0.95,           # GAE smoothing
        clip_range=0.2,            # PPO clipping parameter
        n_epochs=10,               # Number of epochs per update
        n_steps=2048,              # Steps per environment before update
        batch_size=64,            # Larger batch for stable updates with 4 envs
        ent_coef=0.005,            # Lower entropy for more focused policy
        vf_coef=0.5,               # Value function coefficient
        max_grad_norm=0.5,         # Gradient clipping
        policy_kwargs=policy_kwargs,
        seed=42,
        tensorboard_log="./tensorboard_logs/"
    )
    
    print("[INFO] PPO model initialized with improved hyperparameters")

    # 3. Setup Callbacks for Better Training
    # Checkpoint: Save model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=20_000,          # Save every 20k steps
        save_path="./checkpoints/",
        name_prefix="ppo_vpp",
        save_vecnormalize=True     # Save normalization stats with checkpoints
    )
    
    # Evaluation: Track performance on unseen episodes
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./checkpoints/best_model/",
        log_path="./eval_logs/",
        eval_freq=10_000,          # Evaluate every 10k steps for better tracking
        n_eval_episodes=10,        # More episodes for robust evaluation
        deterministic=True,        # Use deterministic policy for evaluation
        render=False,
        verbose=1
    )
    
    # Combine callbacks
    callbacks = CallbackList([checkpoint_callback, eval_callback])
    
    # 4. Start Training
    total_timesteps = 500_000  # Increased to 0.5M for better convergence
    print("[START] PPO Training...")
    print(f"   Target: {total_timesteps:,} Timesteps")
    print(f"   Checkpoints every: {checkpoint_callback.save_freq:,} steps")
    print(f"   Evaluation every: {eval_callback.eval_freq:,} steps")
    print()
    
    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)

    # 5. Save Final Model
    model_name = "ppo_vpp_aggregator"
    model.save(f"./checkpoints/{model_name}")
    env.save(f"./checkpoints/{model_name}_vecnormalize.pkl")
    
    # Also save the best model's normalization stats
    eval_env.save(f"./checkpoints/best_model/vecnormalize.pkl")
    
    print()
    print("[OK] Training Complete!")
    print(f"   Final model: ./checkpoints/{model_name}.zip")
    print(f"   Best model: ./checkpoints/best_model/best_model.zip")
    print(f"   Normalization stats saved")
    print()
    print("To view training progress:")
    print("   tensorboard --logdir=./tensorboard_logs/")

if __name__ == "__main__":
    main()            


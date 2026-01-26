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


def main():
    # Safety: Create directories
    os.makedirs("./checkpoints/", exist_ok=True)
    os.makedirs("./tensorboard_logs/", exist_ok=True)
    os.makedirs("./eval_logs/", exist_ok=True)
    
    print("[INFO] Checking environment validity...")
    check_env(UrbanVPPEnv(data_path="./data"), warn=True)
    
    # 1. Create Training Environments (Multiple for parallel training)
    n_envs = 4  # Use 4 parallel environments for faster training
    def make_env():
        def _init():
            env = Monitor(UrbanVPPEnv(data_path="./data"))
            return env
        return _init
    
    # Use SubprocVecEnv for true parallel execution (faster on multi-core CPUs)
    env = SubprocVecEnv([make_env() for _ in range(n_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=5.0, clip_reward=10.0)
    
    # 2. Create Evaluation Environment (Separate to track actual performance)
    eval_env = DummyVecEnv([make_env()])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=5.0, training=False)
    
    print("[OK] Environment created and normalized.")
    print(f"[INFO] Training with {n_envs} parallel environments")

    # 2. Define the PPO Model
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=3e-4,        # Standard learning rate for PPO
        gamma=0.99,                # Discount factor
        gae_lambda=0.95,           # GAE smoothing
        clip_range=0.2,            # PPO clipping parameter
        n_epochs=10,               # Number of epochs per update
        n_steps=2048,              # Steps per environment before update
        batch_size=64,             # Minibatch size
        ent_coef=0.01,             # Entropy bonus for exploration
        vf_coef=0.5,               # Value function coefficient
        max_grad_norm=0.5,         # Gradient clipping
        seed=42,
        tensorboard_log="./tensorboard_logs/"
    )
    
    print("[INFO] PPO model initialized with improved hyperparameters")

    # 3. Setup Callbacks for Better Training
    # Checkpoint: Save model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=20_000,          # Save every 20k steps (adjusted for parallel envs)
        save_path="./checkpoints/",
        name_prefix="ppo_vpp",
        save_vecnormalize=True     # Save normalization stats with checkpoints
    )
    
    # Evaluation: Track performance on unseen episodes
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./checkpoints/best_model/",
        log_path="./eval_logs/",
        eval_freq=10_000,          # Evaluate every 10k steps
        n_eval_episodes=5,         # Run 5 episodes for evaluation
        deterministic=True,        # Use deterministic policy for evaluation
        render=False,
        verbose=1
    )
    
    # Combine callbacks
    callbacks = CallbackList([checkpoint_callback, eval_callback])
    
    # 4. Start Training
    total_timesteps = 500_000  # Increased for better learning
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


import gymnasium as gym
from stable_baselines3 import PPO # Import the class you just wrote
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import os
from vpp_env import UrbanVPPEnv 


def main():
    # Safety: Create directories
    os.makedirs("./checkpoints/", exist_ok=True)
    os.makedirs("./tensorboard_logs/", exist_ok=True)
    check_env(UrbanVPPEnv(data_path="./data"), warn=True) # Check if the custom environment follows Gymnasium's interface
    # 1. Create the Environment
    def make_env():
        env = Monitor(UrbanVPPEnv(data_path="./data"))  # Monitor to log episode stats
        #env = UrbanVPPEnv()
        return env
    
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=5.0)
    # Reset it once to check if everything works
    #env.reset()
    print("✅ Environment created and normalized.")

    # 2. Define the PPO Model (The "Brain")
    # We use "MlpPolicy" because your inputs are just numbers.
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,                 # Print logs to console
        learning_rate=0.0003,      # Speed of learning (Standard default)
        gamma=0.99,                # Discount Factor: How much it cares about future rewards
        gae_lambda=0.95,           # Smoothing for Advantage calculation
        clip_range=0.1,            # Safety constraint for the Brain update
        n_epochs=10,               # optimize the surrogate loss 10 times
        n_steps=2048,              # Number of steps to run before updating the brain
        batch_size=64,             # Number of samples to look at, at once
        ent_coef=0.01,             # Entropy Coefficient: Encourages exploration
        max_grad_norm=0.5,
        seed=42,
        tensorboard_log="./tensorboard_logs/" # Where to save the graphs
    )

    # 3. Start Training
    print("🚀 Starting PPO Training...")
    print("   Target: 100,000 Timesteps")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path="./checkpoints/",
        name_prefix="ppo_vpp"
    )
    # 100,000 steps is enough to see initial results. 
    # For your final thesis graphs, you might want 500,000 or 1,000,000.
    model.learn(total_timesteps=100_000, callback=checkpoint_callback)

    # 4. Save the Model
    model_name = "ppo_vpp_aggregator"
    model.save(model_name)
    env.save("vecnormalize_stats.pkl")
    print(f"✅ Training Complete! Model saved as '{model_name}.zip'")

if __name__ == "__main__":
    main()            


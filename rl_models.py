# agents/rl_models.py
from stable_baselines3 import PPO, A2C, DQN

def train_rl_model(algo, env, log_dir, total_timesteps=100_000):
    cls = {'PPO': PPO, 'A2C': A2C, 'DQN': DQN}[algo]
    model = cls("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    model.learn(total_timesteps=total_timesteps)
    model.save(f"models/{algo.lower()}_model")
    return model
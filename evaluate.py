# eval/evaluate.py
def evaluate_agent(agent, env, name="Agent", n_episodes=5):
    rewards, response_times = [], []
    for ep in range(n_episodes):
        obs, info = env.reset()
        done, total_reward = False, 0
        while not done:
            action, _ = agent.predict(obs)
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
            for s in info.get('server_states', {}).values():
                rt = s.get('avg_response_time', 0)
                if rt > 0: response_times.append(rt)
        rewards.append(total_reward)
        print(f"{name} - Ep {ep+1}: Reward = {total_reward:.2f}")
    print(f"{name} Mean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"{name} Mean RT: {np.mean(response_times):.6f} s")
    return rewards, response_times
# plots/visualize.py
import matplotlib.pyplot as plt

def plot_agent_comparison(agent_rewards, filename="agent_comparison.png"):
    plt.figure(figsize=(6,4))
    for name, rewards in agent_rewards.items():
        plt.plot(rewards, marker='o', label=name)
    plt.title("Reward Comparison")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()
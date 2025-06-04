# agents/baselines.py
class RoundRobinAgent:
    def __init__(self, num_servers): self.current, self.n = 0, num_servers
    def predict(self, obs, deterministic=True): self.current = (self.current + 1) % self.n; return self.current, None

class LeastConnectionsAgent:
    def __init__(self, env): self.env = env
    def predict(self, obs, deterministic=True): return int(np.argmin([len(s['active_requests']) for s in self.env.servers])), None

class LeastLoadedAgent:
    def __init__(self, env): self.env = env
    def predict(self, obs, deterministic=True): return int(np.argmin([s['current_load_tokens'] for s in self.env.servers])), None
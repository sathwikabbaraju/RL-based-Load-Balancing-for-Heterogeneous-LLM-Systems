# env/load_balancer_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import collections

class LoadBalancingEnv(gym.Env):
    def __init__(self, trace_data, num_servers=3, server_configs=None,
                 reward_weights=None, response_time_history_len=100):
        super().__init__()
        self.trace_data = trace_data.copy().sort_values(by='TIMESTAMP').reset_index(drop=True)
        self.num_servers = num_servers
        self.response_time_history_len = response_time_history_len
        self.reward_weights = reward_weights or {
            'response_time_penalty_per_sec': -0.01,
            'queue_penalty_per_item': -0.1,
            'overload_penalty': -10,
            'utilization_balance_reward_factor': 0.1,
            'request_completion_reward': 1,
            'slo_violation_penalty': -5
        }

        self.server_configs = server_configs or [
            {"rate": 5000, "max_concurrent": 5} for _ in range(self.num_servers)
        ]
        self.current_time = self.trace_data['TIMESTAMP'].iloc[0]
        self.servers = []
        for i in range(self.num_servers):
            cfg = self.server_configs[i]
            self.servers.append({
                'id': i,
                'processing_rate_tps': cfg['rate'],
                'max_concurrent_requests': cfg['max_concurrent'],
                'active_requests': [],
                'queue': collections.deque(),
                'current_load_tokens': 0,
                'last_request_finish_time': self.current_time,
                'processed_requests_count': 0,
                'total_processed_tokens': 0,
                'response_times': collections.deque(maxlen=self.response_time_history_len)
            })

        # Observation & Action space
        self.observation_space = spaces.Box(low=-1, high=1, shape=(10 + 5 * self.num_servers,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.num_servers)
        self.current_request_idx = 0
        self.done = False

    def _get_obs(self, current_request):
        request_features = [
            current_request['TotalTokens'],
            current_request['ContextTokens'],
            current_request['GeneratedTokens'],
            current_request['HourOfDay'],
            current_request['DayOfWeek'],
            current_request['TotalTokens_last_5_mean'],
            current_request['TotalTokens_last_5_std'],
            current_request['TotalTokens_lag_1'],
            current_request['ContextTokens_last_5_mean'],
            current_request['GeneratedTokens_last_5_mean'],
        ]
        server_features = []
        for s in self.servers:
            utilization = len(s['active_requests']) / s['max_concurrent_requests']
            est_wait = sum(req['total_tokens'] for req in s['queue']) / s['processing_rate_tps']
            avg_rt = np.mean(s['response_times']) if s['response_times'] else 0.0
            server_features.extend([
                s['current_load_tokens'],
                len(s['queue']),
                utilization,
                est_wait,
                avg_rt
            ])
        return np.array(request_features + server_features, dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.current_request_idx = 0
        self.current_time = self.trace_data['TIMESTAMP'].iloc[0]
        self.done = False
        for s in self.servers:
            s['active_requests'].clear()
            s['queue'].clear()
            s['current_load_tokens'] = 0
            s['last_request_finish_time'] = self.current_time
            s['processed_requests_count'] = 0
            s['total_processed_tokens'] = 0
            s['response_times'].clear()
        obs = self._get_obs(self.trace_data.iloc[0])
        info = {'server_states': {s['id']: {'active_requests': len(s['active_requests']),
                                            'queue_length': len(s['queue']),
                                            'current_load_tokens': s['current_load_tokens']} for s in self.servers}}
        return obs, info

    def step(self, action):
        req = self.trace_data.iloc[self.current_request_idx]
        self.current_time = req['TIMESTAMP']
        # Update server state...
        # Process request...
        # Apply reward components including SLO penalty if needed...
        # See your existing logic and plug in here.
        # Then return (next_obs, reward, done, truncated, info)
        ...
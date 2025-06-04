# Intelligent Load Balancing Using Reinforcement Learning for Heterogeneous LLM Inference Systems

**Developed by Sai Sathwik Abbaraju**  
Connect with me on LinkedIn: [@sai-sathwik-abbaraju-6527ab275](https://www.linkedin.com/in/sai-sathwik-abbaraju-6527ab275)

---

## Project Overview
I built an end-to-end framework that leverages reinforcement learning (RL) to intelligently route inference requests across a heterogeneous cluster of LLM (Large Language Model) servers. By modeling token-level processing costs, server capacities, and real-world workload traces, I created an RL agent that learns to minimize response time, prevent overload, and balance utilization under both nominal and bursty conditions.

---

## Key Contributions
- **Dataset & Feature Engineering**  
  - Sourced Azure LLM inference traces (Coding and Conversation workloads).  
  - Extracted time-based features (hour of day, day of week, etc.) and computed `TotalTokens = ContextTokens + GeneratedTokens`.  
  - Engineered rolling-window statistics (mean/std over the last 5, 10, and 20 requests) and lag-1 features to capture temporal correlations.  
  - Applied RobustScaler to normalize token-related features, reducing sensitivity to outliers.

- **Custom Gym Environment**  
  - Designed `LoadBalancingEnv` to simulate an LLM inference cluster with heterogeneous servers (distinct TPS and concurrency limits).  
  - Each server tracks current queue length, active token load, and recent response-time history.  
  - At each step, the agent receives a combined state (request features + server states) and chooses a server index as its action.

- **Reward Function Design**  
  - Penalized estimated processing time (`–0.01 × estimated_time`) and queue backlog (`–0.1 × queue_length`).  
  - Applied a heavy overload penalty (`–10`) if the chosen server was at maximum concurrency.  
  - Granted a small balance reward (`+0.1 × utilization_balance_metric`) to encourage even load distribution across servers.  
  - Provided a base completion reward (`+1`) for each request that successfully enters processing.  
  - Under high-load experiments, added an SLO penalty (`–5`) if actual response time exceeded 2 s.

- **RL Algorithms & Baselines**  
  - Trained and compared PPO, A2C, and DQN agents (all using an MLP policy with two hidden layers of size 128).  
  - Implemented four classic baselines: Round Robin, Least Connections, Least Loaded, and Random.  
  - Demonstrated that PPO significantly outperforms all baselines—reducing average response times by up to 35 % and cutting queue lengths by 40 %–50 % under bursty conditions.

- **Experimental Results**  
  - **Nominal Load (Homogeneous Servers):** PPO achieved ~0.85 s average response time versus ~1.5 s for Round Robin, with under 5 % SLO violations.  
  - **High Load (Heterogeneous Servers):** With three servers (3 000 TPS/3 conc., 5 000 TPS/5 conc., 7 000 TPS/7 conc.), PPO limited average latency to ~1.15 s and SLO violations to ~12 %, while baselines exceeded 40 %–60 %.

---

## How I Did It
1. **Data Preparation**  
   - Loaded the Azure traces into Pandas, parsed timestamps, and calculated `TotalTokens`.  
   - Extracted hour/day-of-week features and built rolling/lag windows for contextual history.  
   - Scaled all numerical features with a `RobustScaler` to mitigate long-tail outliers.

2. **Environment Implementation**  
   - Created `LoadBalancingEnv` (subclass of `gym.Env`) to simulate request arrivals, server queues, and token-level processing.  
   - On each step:  
     - Combined request features and server state into a single observation vector.  
     - Agent selects one of the servers as its action.  
     - Environment enqueues or immediately schedules the request based on available concurrency.  
     - Completed requests release token load, record actual response time, and contribute to reward calculation.  

3. **Agent Training & Evaluation**  
   - Configured PPO, A2C, and DQN agents from Stable Baselines3 with consistent hyperparameters (learning rate = 3×10⁻⁴, γ = 0.99).  
   - Trained each agent for 200 000 timesteps on nominal-load data and 400 000 timesteps on high-load heterogeneous scenarios.  
   - Evaluated against four heuristics over multiple episodes, measuring cumulative reward, average response time, average queue length, and SLO violation rate.

4. **Visualization & Analysis**  
   - Generated plots illustrating token distributions, hourly request counts, learning curves, response-time distributions, per-server load trajectories, and SLO violation heatmaps.  
   - Analyzed how PPO dynamically routes large-token requests to high-TPS servers while small requests go to slower servers, resulting in lower overall latency and balanced utilization.

---

## What I Accomplished
- **End-to-End RL Framework:** Designed, implemented, and validated a complete RL pipeline for load balancing LLM inference requests on heterogeneous servers.  
- **Realistic Simulation:** Built a custom Gym environment that models token-level processing, queue dynamics, and concurrency constraints.  
- **Superior Performance:** Showed that PPO outperforms traditional heuristics by large margins under both nominal and bursty load conditions.  
- **In-Depth Analysis:** Provided detailed visualizations and ablation studies to demonstrate the impact of feature engineering, reward shaping, and heterogeneity modeling on performance.

---

**This project was developed by Sai Sathwik Abbaraju.**  
Connect with me on LinkedIn: [@sai-sathwik-abbaraju-6527ab275](https://www.linkedin.com/in/sai-sathwik-abbaraju-6527ab275)  # RL-based-Load-Balancing-for-Heterogeneous-LLM-Systems

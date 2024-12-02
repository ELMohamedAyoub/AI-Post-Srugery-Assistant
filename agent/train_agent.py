"""
Purpose:
Trains the RL agent using the custom environment.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.rehab_env import RehabEnv
from agent.rl_agent import RLAgent
import pickle

# Initialize environment and agent
env = RehabEnv()
agent = RLAgent(state_size=3, action_size=4)

# Training parameters
num_episodes = 100
checkpoint_interval = 10  # Save the Q-table every 10 episodes

for episode in range(1, num_episodes + 1):
    state = env.reset().astype(int)
    total_reward = 0

    while not env.done:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        total_reward += reward

        # Update the Q-table
        agent.learn(state, action, reward, next_state)
        state = next_state

    print(f"Episode {episode}: Total Reward = {total_reward}")

    # Save progress every checkpoint_interval episodes
    if episode % checkpoint_interval == 0:
        with open("q_table.pkl", "wb") as f:
            pickle.dump(agent.q_table, f)
        print(f"Checkpoint: Q-table saved at episode {episode}")

# Save final Q-table
with open("q_table.pkl", "wb") as f:
    pickle.dump(agent.q_table, f)
print("Training complete. Final Q-table saved to q_table.pkl.")

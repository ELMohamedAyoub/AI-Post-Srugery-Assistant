"""
Purpose:
Validate the agent and environment.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
from env.rehab_env import RehabEnv
from agent.rl_agent import RLAgent

def test_environment():
    env = RehabEnv()
    state = env.reset()
    assert len(state) == 3  # State should have three variables

def test_agent_learning():
    agent = RLAgent(state_size=3, action_size=4)
    state = [5, 5, 5]
    action = 0
    reward = 1
    next_state = [6, 5, 5]

    agent.learn(state, action, reward, next_state)
    assert agent.q_table[tuple(state)][action] != 0

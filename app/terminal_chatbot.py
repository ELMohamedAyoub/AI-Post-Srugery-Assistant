"""
Purpose:
Provides the terminal-based interface for interacting with the chatbot.
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

# Load the trained Q-table
try:
    with open("q_table.pkl", "rb") as f:
        agent.q_table = pickle.load(f)
    print("Q-table loaded successfully!")
except FileNotFoundError:
    print("Error: Q-table file not found. Train the agent first by running train_agent.py.")
    exit()
except Exception as e:
    print(f"Error loading Q-table: {e}")
    exit()

# Chatbot actions
actions = {
    0: "Don't forget to take your medication!",
    1: "You're doing amazing! Stay positive!",
    2: "Your next appointment is coming up soon.",
    3: "Here's what you need to know about your knee surgery."
}

def terminal_chatbot():
    state = env.reset().astype(int)

    print("Welcome to your Post-Surgery Assistant!")
    print("I'm here to help with your recovery. Let’s get started!\n")

    while not env.done:
        action = agent.act(state)
        print(f"Chatbot: {actions[action]}")

        feedback = input("How do you feel? (good/neutral/bad): ").strip().lower()
        if feedback == "good":
            state[0] += 1  # Improve mood
        elif feedback == "neutral":
            pass  # No change
        elif feedback == "bad":
            state[0] -= 1
        else:
            print("Invalid input. Please type 'good', 'neutral', or 'bad'.")
            continue

        state, _, done = env.step(action)
        print(f"Current Status: Mood = {state[0]}, Adherence = {state[1]}, Engagement = {state[2]}\n")

    print("Great job! You’ve completed your rehabilitation goals. Take care!")

if __name__ == "__main__":
    terminal_chatbot()

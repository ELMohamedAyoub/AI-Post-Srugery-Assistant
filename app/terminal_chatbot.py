"""
Purpose:
Provides the terminal-based interface for interacting with the chatbot.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
"""
Terminal-based chatbot for post-surgery rehabilitation assistance.
Provides reminders, emotional support, and appointment notifications.
"""

import pickle
import pandas as pd
import datetime
from env.rehab_env import RehabEnv
from agent.rl_agent import RLAgent
import spacy
import csv

# Load NLP model for feedback parsing
nlp = spacy.load("en_core_web_sm")


def load_patient_profile(patient_id):
    """Load a patient's profile from profiles.csv."""
    profiles = pd.read_csv("data/profiles.csv")
    profile = profiles[profiles["patient_id"] == patient_id].to_dict("records")
    if not profile:
        print("Patient profile not found!")
        exit()
    return profile[0]


def parse_feedback(feedback):
    """Parse patient feedback to understand their mood."""
    doc = nlp(feedback)
    if any(token.lemma_ in ["happy", "good", "great"] for token in doc):
        return "good"
    elif any(token.lemma_ in ["okay", "neutral"] for token in doc):
        return "neutral"
    elif any(token.lemma_ in ["sad", "bad", "terrible"] for token in doc):
        return "bad"
    return "neutral"


def check_reminders(patient):
    """Check if it's time for medication or other scheduled tasks."""
    now = datetime.datetime.now().strftime("%I:%M %p")  # 12-hour format
    medication_times = patient["medication_schedule"].split(", ")
    if now in medication_times:
        print("\n🔔 Reminder: It's time to take your medication!\n")


def log_interaction(patient_id, state, action, feedback):
    """Log patient-chatbot interaction to user_sessions.csv."""
    with open("data/user_sessions.csv", "a") as file:
        writer = csv.writer(file)
        writer.writerow([patient_id, state.tolist(), action, feedback])


def terminal_chatbot():
    """Main function to run the terminal chatbot."""
    # Load patient profile
    patient_id = int(input("Enter your patient ID: "))
    patient = load_patient_profile(patient_id)

    print(f"\nWelcome, {patient['name']}! Let's assist you with your recovery.\n")

    # Initialize environment and agent
    env = RehabEnv()
    state = env.reset().astype(int)

    agent = RLAgent(state_size=3, action_size=4)
    with open("q_table.pkl", "rb") as f:
        agent.q_table = pickle.load(f)
    print("Q-table loaded successfully!\n")

    actions = {
        0: "Don't forget to take your medication!",
        1: "You're doing amazing! Stay positive!",
        2: f"Your next appointment is on {patient['appointment_date']}.",
        3: f"Here’s what you need to know about your {patient['surgery_type']}."
    }

    while not env.done:
        check_reminders(patient)

        action = agent.act(state)
        print(f"Chatbot: {actions[action]}")

        feedback = input("How do you feel? (e.g., 'I feel great!'): ").strip()
        parsed_feedback = parse_feedback(feedback)
        if parsed_feedback == "good":
            state[0] += 1
        elif parsed_feedback == "bad":
            state[0] -= 1

        state, reward, done = env.step(action)
        print(f"Current state: Mood={state[0]}, Adherence={state[1]}, Engagement={state[2]}\n")

        log_interaction(patient_id, state, action, parsed_feedback)

    print("🎉 Great job! You’ve completed your rehabilitation goals. Take care!")

# Entry point
if __name__ == "__main__":
    terminal_chatbot()

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
import csv
import spacy
from datetime import datetime
from env.rehab_env import RehabEnv
from agent.rl_agent import RLAgent

# Load NLP model for feedback parsing
nlp = spacy.load("en_core_web_sm")

# Medication schedule (the times should ideally come from the patient's profile)
medication_schedule = {
    "morning": "Painkiller",
    "afternoon": "Anti-inflammatory",
    "evening": "Antibiotic"
}

appointment_date = "2024-12-15"  # Example appointment date

# Check if today is the appointment day
def check_appointment_reminders():
    today = datetime.now().date()
    appointment_day = datetime.strptime(appointment_date, '%Y-%m-%d').date()

    if today == appointment_day:
        print("Don't forget! You have an appointment today.")
    elif (appointment_day - today).days == 1:
        print("Reminder: You have an appointment tomorrow.")

# Function to load patient profile
def load_patient_profile(patient_id):
    """Load a patient's profile from profiles.csv."""
    try:
        profiles = pd.read_csv("data/profiles.csv")
        profile = profiles[profiles["patient_id"] == patient_id].to_dict("records")
        if not profile:
            print("Patient profile not found!")
            exit()
        return profile[0]
    except FileNotFoundError:
        print("Error: Profile data not found.")
        exit()

# Medication check based on time of day
def check_medication_schedule(current_time):
    """Check if it's time for medication based on the current time."""
    if current_time in medication_schedule:
        return f"Time for your {medication_schedule[current_time]}."
    return "No medications right now."

# Function to parse patient feedback to assess mood
def parse_feedback(feedback):
    """Parse patient feedback to determine mood."""
    doc = nlp(feedback)
    if any(token.lemma_ in ["happy", "good", "great", "amazing", "fantastic"] for token in doc):
        return "good"
    elif any(token.lemma_ in ["okay", "neutral", "fine"] for token in doc):
        return "neutral"
    elif any(token.lemma_ in ["sad", "bad", "terrible", "angry"] for token in doc):
        return "bad"
    return "neutral"

# Log the interaction between the patient and the chatbot
def log_interaction(patient_id, state, action, feedback):
    """Log patient-chatbot interaction to user_sessions.csv."""
    with open("data/user_sessions.csv", "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([patient_id, state.tolist(), action, feedback])

# Function to respond to patient's mood with more variety
def respond_to_mood(feedback, mood_state):
    """Generate a response based on patient's mood."""
    if "good" in feedback or "great" in feedback:
        responses = [
            "I'm so glad to hear that! Keep up the good work!",
            "Awesome! Your progress is inspiring, keep going!",
            "Fantastic! You're on the right track, stay positive!"
        ]
        return responses[mood_state[0] % len(responses)]  # Vary the response based on mood level
    elif "bad" in feedback or "sad" in feedback:
        responses = [
            "I'm sorry you're feeling this way. Every small step counts!",
            "It's okay to feel down, but remember you're making progress every day.",
            "I'm here for you. Let's take it one step at a time. You've got this!"
        ]
        return responses[mood_state[0] % len(responses)]  # Vary the response based on mood level
    return "Thank you for sharing! Let’s keep moving forward."

# Main function to run the terminal chatbot
def terminal_chatbot():
    """Main chatbot function."""
    # Load patient profile
    patient_id = int(input("Enter your patient ID: "))
    patient = load_patient_profile(patient_id)

    print(f"\nWelcome, {patient['name']}! Let's assist you with your recovery.\n")

    # Initialize environment and agent
    env = RehabEnv()
    state = env.reset().astype(int)

    agent = RLAgent(state_size=3, action_size=4)

    # Load pre-trained Q-table if available
    try:
        with open("q_table.pkl", "rb") as f:
            agent.q_table = pickle.load(f)
        print("Q-table loaded successfully!\n")
    except FileNotFoundError:
        print("No Q-table found. Starting fresh!\n")

    actions = {
        0: "Don't forget to take your medication!",
        1: "You're doing amazing! Stay positive!",
        2: f"Your next appointment is on {patient['appointment_date']}.",
        3: f"Here’s what you need to know about your {patient['surgery_type']}."
    }

    # Check for appointment reminders    
    check_appointment_reminders()

    while not env.done:
        # Get feedback and mood
        feedback = input("How do you feel? (e.g., 'I feel great!'): ").strip()
        parsed_feedback = parse_feedback(feedback)

        # Update state based on feedback
        if parsed_feedback == "good":
            state[0] += 1  # Increase mood
        elif parsed_feedback == "bad":
            state[0] -= 1  # Decrease mood

        # Select an action from the agent
        action = agent.act(state)
        print(f"\nChatbot: {actions[action]}\n")

        # Log the interaction
        log_interaction(patient_id, state, action, parsed_feedback)

        # Respond to patient's mood
        print(respond_to_mood(feedback, state))

        # Update state and check if done
        state, reward, done = env.step(action)
        print(f"Current state: Mood={state[0]}, Adherence={state[1]}, Engagement={state[2]}\n")

    print("🎉 Great job! You’ve completed your rehabilitation goals. Take care!")

# Daily check-in
def daily_check_in(patient_id):
    """Conduct a daily check-in with the patient."""
    print("Daily Check-In: Please answer the following questions to help us support your recovery better.")

    pain_level = input("On a scale of 1 to 10, how is your pain level today? ")
    energy_level = input("How would you rate your energy level today (1-10)? ")
    rehab_progress = input("On a scale of 1 to 10, how satisfied are you with your rehab progress? ")

    # Log the responses to a file for tracking
    with open("data/checkin_data.csv", "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([patient_id, pain_level, energy_level, rehab_progress, datetime.now().date()])

    # Provide feedback based on answers
    if int(pain_level) > 5:
        print("It seems like your pain level is higher than usual. Would you like to discuss this with your doctor?")
    if int(energy_level) < 5:
        print("You're feeling low on energy today. Make sure to rest and hydrate!")
    if int(rehab_progress) < 5:
        print("It looks like you feel less satisfied with your progress. Let's discuss ways we can improve your rehab plan!")

# Entry point for the chatbot application
if __name__ == "__main__":
    terminal_chatbot()
    patient_id = int(input("Enter your patient ID for daily check-in: "))
    daily_check_in(patient_id)
    check_appointment_reminders()

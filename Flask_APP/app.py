import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import spacy
from datetime import datetime
from env.rehab_env import RehabEnv
from agent.rl_agent import RLAgent
import pickle
import csv
import random
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
#from datasets import load_dataset
#import datasets

#from datasets import load_dataset
app = Flask(__name__, template_folder="../Templates")

# Load NLP model for feedback parsing
nlp = spacy.load("en_core_web_sm")

# Load the AI Medical Chatbot Dataset
#dataset = load_dataset("ruslanmv/ai-medical-chatbot")
#datasets.set_caching_enabled(False)
#datasets.utils.logging.set_verbosity_debug()



df = pd.read_parquet("hf://datasets/ruslanmv/ai-medical-chatbot/dialogues.parquet")


# Medication schedule (the times should ideally come from the patient's profile)
medication_schedule = {
    "morning": "Painkiller",
    "afternoon": "Anti-inflammatory",
    "evening": "Antibiotic"
}

appointment_date = "2024-12-15"  # Example appointment date

def check_appointment_reminders():
    """Check if today or tomorrow is the appointment day."""
    today = datetime.now().date()
    appointment_day = datetime.strptime(appointment_date, '%Y-%m-%d').date()

    if today == appointment_day:
        return "Don't forget! You have an appointment today."
    elif (appointment_day - today).days == 1:
        return "Reminder: You have an appointment tomorrow."
    return None

def load_patient_profile(patient_id):
    """Load a patient's profile from profiles.csv."""
    try:
        profiles = pd.read_csv("data/profiles.csv")
        profile = profiles[profiles["patient_id"] == int(patient_id)].to_dict("records")
        if not profile:
            return None
        return profile[0]
    except FileNotFoundError:
        return None

def check_medication_schedule():
    """Check if it's time for medication based on the current time."""
    current_hour = datetime.now().hour
    if 6 <= current_hour < 12:
        return f"Time for your {medication_schedule['morning']}."
    elif 12 <= current_hour < 18:
        return f"Time for your {medication_schedule['afternoon']}."
    elif 18 <= current_hour < 24:
        return f"Time for your {medication_schedule['evening']}."
    return "No medications right now."

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

def log_interaction(patient_id, state, action, feedback):
    """Log patient-chatbot interaction to user_sessions.csv."""
    with open("data/user_sessions.csv", "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([patient_id, state.tolist(), action, feedback])

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

def send_alert_email(patient_id, heart_rate, blood_pressure, temperature, spo2):
    """Send an alert email to the doctor if the patient's vitals are critical."""
    sender_email = "your_email@example.com"
    receiver_email = "doctor_email@example.com"
    password = "your_email_password"

    message = MIMEMultipart("alternative")
    message["Subject"] = "Critical Patient Alert"
    message["From"] = sender_email
    message["To"] = receiver_email

    text = f"""\
    Patient ID: {patient_id}
    Critical Condition Alert!
    Heart Rate: {heart_rate}
    Blood Pressure: {blood_pressure}
    Temperature: {temperature}
    SpO2: {spo2}
    Please take immediate action.
    """
    part = MIMEText(text, "plain")
    message.attach(part)

    try:
        with smtplib.SMTP_SSL("smtp.example.com", 465) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message.as_string())
    except Exception as e:
        print(f"Error sending email: {e}")

def get_response(query):
    """Get a response from the AI Medical Chatbot Dataset based on the query."""
    for index, row in df.iterrows():
        if query.lower() in row['Patient'].lower():
            return row['Doctor']
    return "I'm sorry, I don't have information on that."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/feedback', methods=['POST'])
def feedback():
    patient_id = request.form['patient_id']
    patient = load_patient_profile(patient_id)
    if not patient:
        return "Patient profile not found!", 404

    # Simulate vital signs
    heart_rate = random.randint(60, 100)
    blood_pressure = f"{random.randint(110, 130)}/{random.randint(70, 90)}"
    temperature = round(random.uniform(36.5, 37.5), 1)
    spo2 = random.randint(95, 100)

    return render_template('feedback.html', patient_id=patient_id, patient_name=patient['name'], surgery_type=patient['surgery_type'], doctor_name=patient['doctor_name'], heart_rate=heart_rate, blood_pressure=blood_pressure, temperature=temperature, spo2=spo2)

@app.route('/chat', methods=['POST'])
def chat():
    patient_id = request.form['patient_id']
    feedback = request.form['feedback']
    heart_rate = request.form['heart_rate']
    blood_pressure = request.form['blood_pressure']
    temperature = request.form['temperature']
    spo2 = request.form['spo2']
    
    patient = load_patient_profile(patient_id)
    if not patient:
        return "Patient profile not found!", 404

    # Initialize environment and agent
    env = RehabEnv("data/dialogues.parquet")
    agent = RLAgent(state_size=3, action_size=4)
    state = env.reset().astype(int)

    agent = RLAgent(state_size=3, action_size=4)

    # Load pre-trained Q-table if available
    try:
        with open("q_table.pkl", "rb") as f:
            agent.q_table = pickle.load(f)
    except FileNotFoundError:
        pass

    actions = {
        0: "Don't forget to take your medication!",
        1: "You're doing amazing! Stay positive!",
        2: f"Your next appointment is on {patient['appointment_date']}.",
        3: f"Here’s what you need to know about your {patient['surgery_type']}."
    }

    # Check for appointment reminders    
    appointment_reminder = check_appointment_reminders()

    parsed_feedback = parse_feedback(feedback)

    # Update state based on feedback
    if parsed_feedback == "good":
        state[0] += 1  # Increase mood
    elif parsed_feedback == "bad":
        state[0] -= 1  # Decrease mood

    # Simulate state update based on vital signs
    critical_condition = False
    if int(heart_rate) > 100 or "high" in blood_pressure or float(temperature) > 37.5 or int(spo2) < 95:
        state[1] -= 1  # Decrease adherence if vitals are not normal
        critical_condition = True

    # Select an action from the agent
    action = agent.act(state)
    chatbot_response = actions[action]

    # Log the interaction
    log_interaction(patient_id, state, action, parsed_feedback)

    # Respond to patient's mood
    mood_response = respond_to_mood(feedback, state)

    # Send alert email if the condition is critical
    if critical_condition:
        send_alert_email(patient_id, heart_rate, blood_pressure, temperature, spo2)
        chatbot_response = "Your vitals are critical. An alert has been sent to your doctor."

    # Get additional response from the dataset
    additional_response = get_response(feedback)

    return render_template('feedback.html', patient_id=patient_id, patient_name=patient['name'], surgery_type=patient['surgery_type'], doctor_name=patient['doctor_name'], heart_rate=heart_rate, blood_pressure=blood_pressure, temperature=temperature, spo2=spo2, chatbot_response=chatbot_response, mood_response=mood_response, appointment_reminder=appointment_reminder, additional_response=additional_response)

@app.route('/appointment')
def appointment():
    return render_template('appointment.html', appointment_date=appointment_date)

@app.route('/recommendations')
def recommendations():
    return render_template('recommendations.html')

if __name__ == "__main__":
    app.run(debug=True)
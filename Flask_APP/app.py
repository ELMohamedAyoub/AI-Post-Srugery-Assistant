import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, render_template, request, redirect, url_for,jsonify

from datetime import datetime
import pandas as pd
import spacy
from datetime import datetime
from env.rehab_env import RehabEnv
from agent.rl_agent import RLAgent
import pickle
import csv
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

app = Flask(__name__, template_folder="../Templates")

# Load NLP model for feedback parsing
nlp = spacy.load("en_core_web_sm")

# Medication schedule (the times should ideally come from the patient's profile)
medication_schedule = {
    "morning": "Painkiller",
    "afternoon": "Anti-inflammatory",
    "evening": "Antibiotic"
}

model_name = "Ellbendls/llama-3.2-3b-chat-doctor"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def load_patient_profile(patient_id):
    try:
        profiles = pd.read_csv("data/profiles.csv")
        profile = profiles[profiles["patient_id"] == int(patient_id)].to_dict("records")
        if not profile:
            return None
        return profile[0]
    except FileNotFoundError:
        return None
    
    
appointment_date = "2024-12-15"  # Example appointment date


def save_vital_signs_to_csv(patient_id, heart_rate, blood_pressure, temperature, spo2):
    with open('data/vital_signs_history.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([patient_id, datetime.now(), heart_rate, blood_pressure, temperature, spo2])
        
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
    try:
        profiles = pd.read_csv("data/profiles.csv")
        profile = profiles[profiles["patient_id"] == int(patient_id)].to_dict("records")
        if not profile:
            return None
        return profile[0]
    except FileNotFoundError:
        return None
    
@app.route('/llama_chat', methods=['POST'])
def llama_chat():
    user_input = request.json.get('input')
    # Add context to the input
    user_input = "User: " + user_input + "\nAssistant: Please provide a detailed response to the user's query.\nAssistant: "
    inputs = tokenizer(user_input, return_tensors='pt')
    # Adjust model parameters
    outputs = model.generate(**inputs, max_length=1000, temperature=0.7, top_p=0.9)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return jsonify({'response': response})


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
    positive_responses = [
        "I'm so glad to hear that! Keep up the good work!",
        "Awesome! Your progress is inspiring, keep going!",
        "Fantastic! You're on the right track, stay positive!",
        "Great to hear! You're doing an amazing job!",
        "Wonderful! Your hard work is paying off!"
    ]

    neutral_responses = [
        "Thank you for sharing! Let’s keep moving forward.",
        "It's good to hear from you. Let's continue making progress.",
        "Thanks for the update. Keep up the steady progress!",
        "I appreciate your feedback. Let's keep going!",
        "Thanks for letting me know. Stay consistent!"
    ]

    negative_responses = [
        "I'm sorry you're feeling this way. Every small step counts!",
        "It's okay to feel down, but remember you're making progress every day.",
        "I'm here for you. Let's take it one step at a time. You've got this!",
        "Don't be discouraged. Every effort you make is valuable.",
        "It's tough, but you're stronger than you think. Keep pushing forward!"
    ]

    very_positive_responses = [
        "You're absolutely crushing it! Keep up the fantastic work!",
        "Amazing! Your dedication is truly inspiring!",
        "You're on fire! Keep up the incredible progress!",
        "Outstanding! Your efforts are making a huge difference!",
        "You're doing phenomenal work! Keep shining!"
    ]

    very_negative_responses = [
        "I'm really sorry you're feeling this way. Remember, it's okay to ask for help.",
        "It's tough right now, but you're not alone. We're here to support you.",
        "I know things are hard, but don't give up. Better days are ahead.",
        "It's okay to feel this way. Take it one day at a time, and don't hesitate to reach out for support.",
        "I'm here for you. Let's work through this together, step by step."
    ]

    if "very good" in feedback or "excellent" in feedback:
        return very_positive_responses[mood_state[0] % len(very_positive_responses)]
    elif "good" in feedback or "great" in feedback:
        return positive_responses[mood_state[0] % len(positive_responses)]
    elif "neutral" in feedback or "okay" in feedback:
        return neutral_responses[mood_state[0] % len(neutral_responses)]
    elif "bad" in feedback or "sad" in feedback:
        return negative_responses[mood_state[0] % len(negative_responses)]
    elif "very bad" in feedback or "terrible" in feedback:
        return very_negative_responses[mood_state[0] % len(very_negative_responses)]
    return "Thank you for sharing! Let’s keep moving forward."

def send_alert_email(patient_id, heart_rate, blood_pressure, temperature, spo2):
    """Send an alert email to the doctor if the patient's vitals are critical."""
    sender_email = ""
    receiver_email = ""
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

def load_vital_signs_history(patient_id):
    try:
        history = pd.read_csv('data/vital_signs_history.csv')
        patient_history = history[history['patient_id'] == int(patient_id)]
        return patient_history
    except FileNotFoundError:
        return pd.DataFrame()  # Return an empty DataFrame if the file doesn't exist
@app.route('/')
def index():    
    return render_template('index.html')
@app.route('/llama_chat_ui')
def llama_chat_ui():
    return render_template('llama_chat.html')

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

    # Save vital signs to CSV
    save_vital_signs_to_csv(patient_id, heart_rate, blood_pressure, temperature, spo2)

    return render_template('feedback.html', patient_id=patient_id, patient_name=patient['name'], surgery_type=patient['surgery_type'], heart_rate=heart_rate, blood_pressure=blood_pressure, temperature=temperature, spo2=spo2)
@app.route('/dashboard/<patient_id>')
def dashboard(patient_id):
    patient = load_patient_profile(patient_id)
    if not patient:
        return "Patient profile not found!", 404

    # Load vital signs history
    history = load_vital_signs_history(patient_id)

    # Check for appointment reminders
    appointment_reminder = check_appointment_reminders()

    # Check medication schedule
    medication_message = check_medication_schedule()

    # Debugging lines
    print("Patient:", patient)
    print("History:", history)
    print("Appointment Reminder:", appointment_reminder)
    print("Medication Message:", medication_message)

    return render_template('dashboard.html', patient=patient, history=history.to_dict('records'), appointment_reminder=appointment_reminder, medication_message=medication_message)
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
    env = RehabEnv()
    state = env.reset().astype(int)

    agent = RLAgent(state_size=3, action_size=15)

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
        3: f"Here’s what you need to know about your {patient['surgery_type']}.",
        4: "Remember to stay hydrated and drink plenty of water.",
        5: "Make sure to get enough rest. Sleep is crucial for your recovery.",
        6: "Keep up with your physical therapy exercises. Consistency is key!",
        7: "If you have any concerns, don't hesitate to contact your healthcare provider.",
        8: "Take a moment to relax and practice deep breathing. It can help reduce stress.",
        9: "You're making great progress! Keep tracking your vital signs regularly.",
        10: "Stay positive and keep a journal of your recovery journey. It can be very motivating.",
        11: "Remember to eat a balanced diet to support your healing process.",
        12: "It's important to follow your doctor's advice closely for the best recovery.",
        13: "If you're feeling down, reach out to a friend or family member for support.",
        14: "Keep an eye on any symptoms and report anything unusual to your doctor.",
        15: "You're not alone in this. We're here to support you every step of the way."
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

    return render_template('feedback.html', patient_id=patient_id, heart_rate=heart_rate, blood_pressure=blood_pressure, temperature=temperature, spo2=spo2, chatbot_response=chatbot_response, mood_response=mood_response, appointment_reminder=appointment_reminder)

@app.route('/appointment')
def appointment():
    return render_template('appointment.html', appointment_date=appointment_date)

@app.route('/history/<patient_id>')
def history(patient_id):
    history = load_vital_signs_history(patient_id)
    return render_template('history.html', history=history.to_dict('records'))
@app.route('/doctor_dashboard')
def doctor_dashboard():
    # Load all patient profiles
    try:
        profiles = pd.read_csv("data/profiles.csv")
    except FileNotFoundError:
        return "No patient profiles found!", 404

    # Load vital signs history
    try:
        history = pd.read_csv("data/vital_signs_history.csv")
    except FileNotFoundError:
        history = pd.DataFrame()  # Empty DataFrame if no history found

    return render_template('doctor_dashboard.html', profiles=profiles.to_dict('records'), history=history.to_dict('records'))

@app.route('/recommendations/<patient_id>')
def recommendations(patient_id):
    patient = load_patient_profile(patient_id)
    if not patient:
        return "Patient profile not found!", 404

    return render_template('recommendations.html', patient_id=patient_id, patient_name=patient['name'], surgery_type=patient['surgery_type'])

if __name__ == "__main__":
    app.run(debug=False)
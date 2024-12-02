# RL-Chatbot-Assistant

A reinforcement learning-based chatbot assistant designed to support post-surgery patients by providing medication reminders, motivational support, and emotional check-ins.

---

## 📂 Project Structure

### **Main Folders and Files**

```plaintext
RL-Chatbot-Assistant/
│
├── app/                   # Core application logic
│   ├── templates/         # HTML templates (for Flask web app)
│   │   └── index.html     # Main chatbot interface
│   ├── static/            # Static assets for web UI
│   │   ├── css/           # Stylesheets
│   │   ├── js/            # JavaScript files
│   │   └── img/           # Images and icons
│   ├── __init__.py        # Flask app initializer
│   ├── routes.py          # API routes for chatbot interaction
│   └── chatbot.py         # Chatbot logic interface for RL agent
│
├── model/                 # Machine learning and RL models
│   ├── rl_environment.py  # Custom Gym environment for RL training
│   ├── rl_agent.py        # RL agent implementation
│   ├── train_agent.py     # Training script for the RL agent
│   └── saved_model.pkl    # Trained model saved for use
│
├── data/                  # Data storage and logging
│   ├── user_sessions.csv  # Interaction logs for training insights
│   └── feedback.csv       # Patient feedback for improving the agent
│
├── tests/                 # Test cases for validation
│   ├── test_environment.py  # Unit tests for RL environment
│   ├── test_agent.py        # Unit tests for RL agent
│   └── test_routes.py       # Integration tests for Flask routes
│
├── main.py                # Entry point to run the application
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

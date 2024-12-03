import numpy as np

class RehabEnv:
    def __init__(self):
        self.state = np.array([0, 0, 0])  # Mood, Adherence, Engagement
        self.done = False
        self.thresholds = {"mood": 10, "adherence": 8, "engagement": 7}

    def reset(self):
        """Resets the environment to the initial state."""
        self.done = False
        self.state = np.array([0, 0, 0])  # Reset state
        return self.state

    def step(self, action):
        """
        Simulates a chatbot interaction.
        Action: 0 - Medication Reminder, 1 - Emotional Support,
                2 - Appointment Reminder, 3 - Surgery FAQ
        """
        reward = 0
        mood_change = np.random.choice([-1, 0, 1])
        adherence_change = 0
        engagement_change = 0

        if action == 0:  # Medication Reminder
            adherence_change = 1
            reward = 2 if self.state[1] < 10 else 0
        elif action == 1:  # Emotional Support
            mood_change += 1
            reward = 1
        elif action == 2:  # Appointment Reminder
            engagement_change = 1
            reward = 1
        elif action == 3:  # Surgery FAQ
            engagement_change = 1
            mood_change = 1
            reward = 2

        # Update state with clipping to ensure it stays within bounds
        self.state = np.clip(self.state + np.array([mood_change, adherence_change, engagement_change]), 0, 10)

        # Check if all state values are at the goal (10)
        if all(self.state >= 10):
            self.done = True
            reward += 5  # Bonus reward for completing all goals

        return self.state, reward, self.done

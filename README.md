---

# AI Project - Flask Version

This repository is designed for the project we will be doing in the AI class, focusing on developing a Flask-based web application for a Reinforcement Learning (RL) chatbot assistant for post-surgery rehabilitation patients.

## Table of Contents

- [Introduction](#introduction)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

Welcome to the Flask Version of the AI Project. This project aims to build a Reinforcement Learning (RL) chatbot assistant to help post-surgery rehabilitation patients with medication reminders, emotional support, appointment reminders, and answers to frequently asked questions.

## Technologies Used

- **Python**: The primary programming language used for developing the AI models and scripts.
- **Flask**: A micro web framework for Python to develop the web application.
- **HTML**: Used for creating web interfaces to interact with the AI models.
- **Jupyter Notebook**: For interactive data exploration and visualization.
- **Libraries**:
  - `numpy`: For numerical operations.
  - `pytest`: For testing.
  - `matplotlib`: For plotting and visualization.
  - `spacy`: For natural language processing.
  - `pandas`: For data manipulation and analysis.

## Project Structure

The repository is organized as follows:

```
AI-Project/
├── env/                   # Custom environment for RL
│   └── rehab_env.py       # Simulates patient-chatbot interactions
├── agent/                 # RL agent implementation
│   ├── rl_agent.py        # Q-learning agent
│   └── train_agent.py     # Training script
├── app/                   # Terminal interface
│   └── terminal_chatbot.py # Main terminal-based chatbot code
├── data/                  # Placeholder for patient interaction data
│   └── user_sessions.csv  # Logs of interactions (future use)
├── tests/                 # Unit tests
│   └── test_agent.py      # Validate RL agent
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## Installation

To get started with the project, follow these steps:

1. Clone the repository:

   ```sh
   git clone https://github.com/ELMohamedAyoub/AI-Project.git
   ```

2. Navigate to the project directory:

   ```sh
   cd AI-Project
   ```

3. Create a virtual environment:

   ```sh
   python -m venv venv
   source venv/bin/activate   # For Linux/Mac
   venv\Scripts\activate      # For Windows
   ```

4. Install the required dependencies:

   ```sh
   pip install -r requirements.txt
   ```

5. Download the Spacy language model:

   ```sh
   python -m spacy download en_core_web_sm
   ```

## Usage

After installing the necessary dependencies, you can start using the project. Below are some common commands:

1. Run the terminal-based chatbot:

   ```sh
   python app/terminal_chatbot.py
   ```

2. Open the Jupyter Notebooks:

   ```sh
   jupyter notebook
   ```

## Contributing

We welcome contributions to this project. To contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

For any questions or inquiries, please contact the project maintainer:

- GitHub: [ELMohamedAyoub](https://github.com/ELMohamedAyoub)

---

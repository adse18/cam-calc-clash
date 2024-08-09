# Arithmetic Quiz Web Application: 🔥Cam Calc Clash🔥

## Overview
Welcome to the Arithmetic Quiz Web Application! This interactive app helps you test your arithmetic skills with a unique twist. To play, you'll need to use your webcam to hold up different objects representing answers to arithmetic questions. Your goal is to get as many points as possible within 60 seconds by correctly identifying the object that matches the correct answer.
The app uses a YOLO model to detect items in the webcam video.

## Features
- **Webcam Integration:** Detects objects held up in front of your webcam.
- **Dynamic Quizzes:** Generates random arithmetic questions.
- **Real-Time Scoring:** Earn points by holding up the correct object.
- **Time Limit:** Complete as many questions as you can within 60 seconds.

## Installation Instructions

### 1. Create a Virtual Environment
Download the code, then navigate to your project directory and create a virtual environment first:
```bash
python -m venv venv_name
```

### 2. Activate the Virtual Environment
On Windows, activate the virtual environment with:
```bash
venv_name\Scripts\activate
```

### 3. Install Required Packages
With the virtual environment activated navigate to the project code directory and install the required packages from `requirements.txt`:
```bash
cd cam-calc-clash
```
```bash
pip install -r requirements.txt
```

### 4. Run the Application
Execute the application:
```bash
python run.py
```

### 5. Access the Web Application
Open your web browser and go to the following URL:
```bash
http://127.0.0.1:5000
```

### 6. How to Play
Start the Video Feed: Click the "Start Video" button to begin the webcam feed.
Start the Quiz: Click the "Start Quiz" button to begin answering questions.
Hold Up Objects: When prompted with a question, hold up one of the following objects:
- Scissors for the first answer option.
- Cup for the second answer option.
- Cell Phone for the third answer option.

Score Points: If the detected object matches the correct answer, you earn a point.
Repeat: Continue answering questions until the timer runs out.

### 7. Dependencies
- Flask
- OpenCV
- PyTorch
- Other dependencies are listed in requirements.txt

### 8. Troubleshooting
- Ensure your webcam is properly connected and accessible. Object detection works also better in front of a blanc background, so for the best experience sit in front of a wall when playing.
- Check that all required packages are installed and up-to-date.

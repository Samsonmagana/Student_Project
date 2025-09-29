# BA 3.2 Groupe 
#Project Documentation – Car Agent AI

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import scrolledtext
import pyttsx3

# === Voice configuration ===
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# === Load dataset ===
DATA_PATH = "car_problems.csv"
df = pd.read_csv(DATA_PATH)

# Normalize column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Category mapping (group similar solutions together)
mapping = {
    "Brake pad replacement": "Brake service",
    "Brake fluid replacement": "Brake service",
    "Brake replacement": "Brake service",
    "Tire replacement": "Tire service",
    "Tire rotation": "Tire service",
    "Tire rotation and balancing": "Tire service",
    "Headlight replacement": "Headlight service",
    "Headlight restoration": "Headlight service",
    "Oil filter replacement": "Oil service",
    "Oil replacement": "Oil service",
    "Oil seal replacement": "Oil service",
    "Oil leak": "Oil service",
    "Body repair": "Body repair",
    "Body repair and repaint": "Body repair",
    "Dents and scratches": "Body repair",
}
df["solution_used"] = df["solution_used"].replace(mapping)

# Features and labels
X = df["common_problem"].astype(str)
y = df["solution_used"].astype(str)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=5000)
model.fit(X_train_vec, y_train)

# Evaluate model
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print("=== Model Results ===")
print("Accuracy:", round(accuracy * 100, 2), "%")

# === Car Agent function ===
def car_agent_response(query):
    query_vec = vectorizer.transform([query])
    prediction = model.predict(query_vec)
    return prediction[0]

# Typing effect for chatbot replies
def typewriter_effect(message):
    chat_area.configure(state='normal')
    chat_area.insert(tk.END, "Car Agent: ", "agent")
    chat_area.see(tk.END)
    chat_area.update()
    for char in message:
        chat_area.insert(tk.END, char)
        chat_area.see(tk.END)
        chat_area.update()
        chat_area.after(20)  # 20 ms delay per letter
    chat_area.insert(tk.END, "\n\n")
    chat_area.configure(state='disabled')
    speak(message)

# Handle user question
def ask_question():
    user_question = entry.get()
    if user_question.strip() == "":
        return
    chat_area.configure(state='normal')
    chat_area.insert(tk.END, f"You: {user_question}\n", "user")
    chat_area.tag_add("right", "end-2l linestart", "end-2l lineend")
    chat_area.configure(state='disabled')

    response = car_agent_response(user_question)
    typewriter_effect(response)
    entry.delete(0, tk.END)

# Exit application
def exit_app():
    root.destroy()


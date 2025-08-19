import tkinter as tk
from tkinter import scrolledtext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import string
import nltk

# ---------------------------
# Auto-download NLTK resources if missing
# ---------------------------
def download_nltk_data():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab")

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

download_nltk_data()

# ---------------------------
# Load FAQ data from CSV
# ---------------------------
try:
    faq_df = pd.read_csv("FAQ.csv")
except FileNotFoundError:
    print("Error: FAQ.csv not found in the current directory.")
    exit()

# ---------------------------
# NLP Preprocessing Function
# ---------------------------
def preprocess(text):
    text = str(text).lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    return " ".join(tokens)

# Preprocess FAQ questions
faq_df['processed_question'] = faq_df['question'].apply(preprocess)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(faq_df['processed_question'])

# ---------------------------
# Function to find best match
# ---------------------------
def get_best_answer(user_query):
    user_query_processed = preprocess(user_query)
    user_vector = vectorizer.transform([user_query_processed])
    similarities = cosine_similarity(user_vector, faq_vectors)
    best_match_index = similarities.argmax()
    return faq_df.iloc[best_match_index]['answer']

# ---------------------------
# Tkinter Chatbot UI
# ---------------------------
def send_message():
    user_input = entry.get("1.0", tk.END).strip()
    if user_input == "":
        return
    
    chat_window.config(state=tk.NORMAL)
    chat_window.insert(tk.END, "🧑 You: " + user_input + "\n", "user")
    
    bot_response = get_best_answer(user_input)
    chat_window.insert(tk.END, "🤖 Bot: " + bot_response + "\n\n", "bot")
    
    chat_window.config(state=tk.DISABLED)
    entry.delete("1.0", tk.END)

# Main window
root = tk.Tk()
root.title("💬 FAQ Chatbot")
root.geometry("600x600")
root.configure(bg="#f0f0f5")

# Chat display
chat_window = scrolledtext.ScrolledText(root, bg="white", fg="black", wrap=tk.WORD, font=("Arial", 12))
chat_window.config(state=tk.DISABLED)
chat_window.tag_config("user", foreground="blue", font=("Arial", 12, "bold"))
chat_window.tag_config("bot", foreground="green", font=("Arial", 12))
chat_window.pack(padx=15, pady=15, fill=tk.BOTH, expand=True)

# Frame for input and button
bottom_frame = tk.Frame(root, bg="#f0f0f5")
bottom_frame.pack(fill=tk.X, padx=15, pady=10)

# Entry box (multi-line, bigger)
entry = tk.Text(bottom_frame, height=3, bg="white", fg="black", font=("Arial", 12))
entry.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

# Send button
send_btn = tk.Button(bottom_frame, text="Send ➤", bg="#4CAF50", fg="white", font=("Arial", 12, "bold"), command=send_message)
send_btn.pack(side=tk.RIGHT)

# Run the app
root.mainloop()

# Streamlit ChatBot Web App (Enhanced Version)

import streamlit as st
import json
import string
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import nltk

nltk.download('punkt')

# Load your dataset
with open("realdata.json", "r") as file:
    data = json.load(file)

# Normalize text
def normalize(text):
    return text.lower().translate(str.maketrans('', '', string.punctuation))

# Prepare data
questions = list(data.keys())
answers = list(data.values())
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

# Chat function
def get_bot_response(user_input):
    user_input_clean = normalize(user_input)
    input_vec = vectorizer.transform([user_input_clean])
    similarity = cosine_similarity(input_vec, X)
    index = similarity.argmax()
    score = similarity[0][index]
    return random.choice(answers[index]) if score > 0.3 else "Hmm... I’m not sure how to respond 🤔"

# Streamlit page setup
st.set_page_config(page_title="ChatPy Web", layout="centered")
st.title("🤖 ChatPy - Web Chatbot")
st.write("Type a message and ChatPy will respond intelligently.")

# Session state for history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input form to allow Enter key submission
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("You:", "", key="input_text")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    reply = get_bot_response(user_input)
    timestamp = datetime.now().strftime("%H:%M")
    st.session_state.chat_history.append((user_input, reply, timestamp))

# Optional: Clear chat history
if st.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.experimental_rerun()

# Optional: Download chat
if st.download_button("⬇️ Download Chat", data="\n\n".join([f"You ({t}): {u}\nBot ({t}): {b}" for u, b, t in st.session_state.chat_history]), file_name="chat_history.txt"):
    st.success("Downloaded chat history!")

# Display conversation (top-down order)
for user, bot, time in st.session_state.chat_history:
    st.markdown(f"<div style='background-color:#e1f5fe;padding:10px;border-radius:10px;margin-bottom:5px'><b>You ({time}):</b> {user}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='background-color:#f1f8e9;padding:10px;border-radius:10px;margin-bottom:15px'><b>Bot ({time}):</b> {bot}</div>", unsafe_allow_html=True)

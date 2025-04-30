# Streamlit ChatBot Web App (based on your existing logic)

import streamlit as st
import json
import string
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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

# Input box
user_input = st.text_input("You:", "", key="input")

if st.button("Send") and user_input:
    reply = get_bot_response(user_input)
    st.session_state.chat_history.append((user_input, reply))

# Display conversation
for i, (user, bot) in enumerate(reversed(st.session_state.chat_history)):
    st.markdown(f"**You:** {user}")
    st.markdown(f"**Bot:** {bot}")

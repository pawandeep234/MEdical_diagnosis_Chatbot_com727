from sentence_transformers import SentenceTransformer

sentences_model = SentenceTransformer('all-MiniLM-L6-v2')

import numpy as np
import streamlit as st

from keras.models import load_model

model = load_model('models/model.h5')
print(model.summary())
import json
import random

data = json.loads(open('data/data.json').read())


def predict(input):
    input_encodings = sentences_model.encode(input)
    input_encodings = np.array(input_encodings).reshape(1, -1)
    print(input_encodings.shape)
    prediction = model.predict(input_encodings, batch_size=1)
    prediction_idx = np.argmax(prediction, axis=1)
    if np.max(prediction[0]) < 0.5:
        return "I'm sorry, I don't understand your question."
    return data['responses'][prediction_idx[0]]




# Set Streamlit page config
st.set_page_config(page_icon='🤖', page_title="Medical_Diagnostic_Chatbot", layout='wide')

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Get user input
user_input = st.chat_input("Hello! How may I help you?")
if user_input:
    st.session_state.chat_history.append({'user': 'user', 'text': user_input})
    response = predict(user_input)
    st.session_state.chat_history.append({'user': 'assistant', 'text': response})

# Display chat history
with st.container():
    for data in st.session_state.chat_history:
        _, left, _, right = st.columns((1, 1, 1, 2))
        if data['user'] == 'user':
            with left:
                st.chat_message(data['user']).write(data['text'])
        else:
            with right:
                st.chat_message(data['user']).write(data['text'])

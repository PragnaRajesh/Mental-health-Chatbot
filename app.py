import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import nltk
nltk.download('popular')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import pickle
import numpy as np
from keras.models import load_model
from transformers import pipeline
import json
import random

# Load your Keras model
model = load_model('model.h5')

# Load intents and other necessary data
intents = json.loads(open('intents.json').read())
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))

# Function to clean up user input for classification
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Function to create bag of words representation for the input
def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)  
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return np.array(bag)

# Function to predict the intent/class of the user input
def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# Function to get a response based on the predicted intent
def getResponse(ints, intents_json):
    if ints: 
        tag = ints[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
        return result
    else:
        return "Sorry, I didn't understand that."

# Function to handle user messages and generate a chatbot response
def chatbot_response(msg):
    res = getResponse(predict_class(msg, model), intents)
    return res

# Flask setup
from flask import Flask, render_template, request
app = Flask(__name__)
app.static_folder = 'static'

# Route to render the index.html page
@app.route("/")
def home():
    return render_template("index.html")

# Route to handle AJAX requests from the frontend
@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    print("User Input:", userText)

    # Determine language of user input (currently simplified)
    detected_language = 'en'  # Assume English for simplicity

    # Generate bot response based on user input
    chatbot_response_text = chatbot_response(userText)
    print("Bot Response:", chatbot_response_text)

    return chatbot_response_text

if __name__ == "__main__":
    app.run()

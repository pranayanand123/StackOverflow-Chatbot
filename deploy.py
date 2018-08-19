from flask import Flask, render_template, request
from dialogue_manager import *
from utils import *


app = Flask(__name__)

chatbot = DialogueManager(RESOURCE_PATH)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return str(chatbot.generate_answer(userText))


if __name__ == "__main__":
    app.run()
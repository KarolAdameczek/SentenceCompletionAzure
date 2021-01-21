from flask import Flask, render_template, request
import requests
app = Flask(__name__)

def getSentences(text):
    # to be implemented
    # send request to the ML model for new sentences

    sentences = []

    for i in range(10):
        sentences.append(text + " " + str(i))

    return sentences

@app.route("/")
def hello():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    try:
        json = request.get_json()
        text = json["text"]
    except:
        return ""

    return {"sentences" : getSentences(text)}

if __name__ == "__main__":
    app.run()
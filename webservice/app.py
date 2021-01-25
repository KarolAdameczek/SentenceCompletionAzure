from flask import Flask, render_template, request
from dotenv import load_dotenv
from os import getenv
import requests
app = Flask(__name__)

load_dotenv()
PUNCTUATION = ",.?:;…"
NUM = 5

MODELS = {
    1 : {
        "text" : "Model tworzący zdania litera po literze",
        "url" : getenv("MODEL1_URL"),
        "token" : getenv("MODEL1_TOKEN")
    }
}

def getSentences(text, num, model):
    data = {"data" : text.lower(), "num" : num}
    sentences = requestSentences(data, model)
    return sentences

def requestSentences(data, model):
    headers = {'Authorization': 'Bearer ' + model["token"]}
    res = requests.post(model["url"], headers=headers, json=data)
    
    texts = res.json()["data"]
    for j in range(len(texts)):
        i = 0
        while i < len(texts[j]):
            if texts[j][i] in PUNCTUATION and texts[j][i - 1] == " ":
                texts[j] = texts[j][:i - 1 ] + texts[j][i:]
            else:
                i += 1
    return texts

@app.route("/")
def hello():
    return render_template("index.html", models=MODELS)

@app.route("/generate", methods=["POST"])
def generate():
    try:
        json = request.get_json()
        text = json["text"]
        model = int(json["model"])
        num = int(json["num"]) if json["num"] != "" else 5

        if model not in MODELS:
            return ""

        if num < 1 or num > 20:
            return ""
    except:
        return ""

    return {"sentences" : getSentences(text, num, MODELS[model])}

if __name__ == "__main__":
    app.run()
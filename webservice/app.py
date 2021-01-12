from flask import Flask, render_template
app = Flask(__name__)

@app.route("/")
def hello():
    return render_template("index.html")

@app.route("/generate")
def generate():
    return {
        "message" : "Nothing here yet ¯\\_(ツ)_/¯"
    }

if __name__ == "__main__":
    app.run(debug=True)
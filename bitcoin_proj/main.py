from ai_model import arima_model
from flask import Flask, render_template, g
import importlib

app = Flask(__name__)

@app.route('/')
def hello():
    accuracy = arima_model() * 100
    accuracy = "{:.2f}".format(accuracy)
    return render_template("home.html", accuracy=accuracy)
    
if __name__ == "__main__":
    app.run(debug=True)
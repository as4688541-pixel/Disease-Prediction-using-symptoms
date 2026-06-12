from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

with open("model/disease_model.pkl", "rb") as f:
    model = pickle.load(f)

symptoms = pd.read_csv("dataset/Training.csv").drop("prognosis", axis=1).columns.tolist()

@app.route("/")
def index():
    return render_template("index.html", symptoms=symptoms)

@app.route("/predict", methods=["POST"])
def predict():
    input_symptoms = request.form.getlist("symptoms")
    input_vector = [1 if symptom in input_symptoms else 0 for symptom in symptoms]
    prediction = model.predict([input_vector])[0]
    return render_template("result.html", prediction=prediction, selected=input_symptoms)

if __name__ == "__main__":
    app.run(debug=True)

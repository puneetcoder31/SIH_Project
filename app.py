from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained Random Forest model
model = pickle.load(open("career_model.pkl", "rb"))

# -------------------- ROUTES --------------------

# Home page (dashboard with "Start Quiz" button)
@app.route("/")
def home():
    return render_template("index.html")

# Quiz page (actual quiz form)
@app.route("/quiz")
def quiz():
    return render_template("quiz.html")

# Predict endpoint (form submission from quiz.html)
@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # ---- Inputs from quiz.html ----
        math = int(request.form["Math"])
        science = int(request.form["Science"])
        business = int(request.form["Business"])
        creativity = int(request.form["Creativity"])
        communication = int(request.form["Communication"])

        # Convert to numpy array for model
        user_input = np.array([[math, science, business, creativity, communication]])

        # Predict stream/college
        prediction = model.predict(user_input)[0]

        # Optional mapping (you can map your model outputs to friendly text)
        streams = {
            "Science - Govt College": "Science - PCM (Good for Engineering/AI)",
            "Science - Private College": "Science - PCB (Good for Medical/Biology)",
            "Commerce - Govt College": "Commerce (Business, Finance, CA/CS)",
            "Commerce - Private College": "Commerce (Business, Finance, CA/CS)",
            "Arts - Tier-1 College": "Arts/Humanities (UPSC, Law, Social Sciences)",
            "Arts - Govt College": "Arts/Humanities (UPSC, Law, Social Sciences)",
            "Science - Tier-1 College": "Science - PCM (Good for Engineering/AI)",
            "Commerce - Tier-1 College": "Commerce (Business, Finance, CA/CS)",
            "Arts - Private College": "Arts/Humanities (UPSC, Law, Social Sciences)"
        }

        colleges = {
            "Science - Govt College": "Top Govt Engineering/Science Colleges",
            "Science - Private College": "Good Private Science Colleges",
            "Commerce - Govt College": "Top Govt Commerce Colleges",
            "Commerce - Private College": "Good Private Commerce Colleges",
            "Arts - Tier-1 College": "Tier-1 Arts/Humanities Colleges",
            "Arts - Govt College": "Government Arts Colleges",
            "Science - Tier-1 College": "Tier-1 Science Colleges",
            "Commerce - Tier-1 College": "Tier-1 Commerce Colleges",
            "Arts - Private College": "Private Arts Colleges"
        }

        result_stream = streams.get(prediction, "General Guidance")
        result_college = colleges.get(prediction, "Nearby State University")

        # Render result page
        return render_template("result.html",
                               stream=result_stream,
                               college=result_college,
                               math=math,
                               science=science,
                               business=business,
                               creativity=creativity,
                               communication=communication)

    return "Error: Invalid Request"


if __name__ == "__main__":
    app.run(debug=True)

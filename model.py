import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# ----------------------------
# Step 1: Create Sample Dataset
# ----------------------------
# Features: [Math, Science, Business, Creativity, Communication]
# Target: Stream/College
data = {
    "Math": [9, 8, 2, 5, 1, 3, 7, 4, 6, 2],
    "Science": [8, 9, 1, 6, 2, 4, 7, 3, 5, 1],
    "Business": [2, 3, 9, 5, 8, 7, 4, 6, 1, 9],
    "Creativity": [4, 3, 7, 8, 9, 6, 5, 2, 4, 8],
    "Communication": [3, 4, 6, 7, 8, 9, 5, 2, 4, 7],
    "Stream": [
        "Science - Govt College",
        "Science - Private College",
        "Commerce - Govt College",
        "Commerce - Private College",
        "Arts - Tier-1 College",
        "Arts - Govt College",
        "Science - Tier-1 College",
        "Commerce - Tier-1 College",
        "Arts - Private College",
        "Commerce - Govt College",
    ],
}

df = pd.DataFrame(data)

# ----------------------------
# Step 2: Prepare Data
# ----------------------------
X = df.drop("Stream", axis=1)
y = df["Stream"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------
# Step 3: Train Random Forest
# ----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ----------------------------
# Step 4: Save Model
# ----------------------------
with open("career_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as career_model.pkl")

# ----------------------------
# Step 5: Ask User for Input
# ----------------------------
print("\nAnswer the following questions (scale 1-10):")
math = int(input("How much do you like Mathematics (1-10)? "))
science = int(input("How much do you like Science (1-10)? "))
business = int(input("How much do you like Business/Commerce (1-10)? "))
creativity = int(input("How creative are you (1-10)? "))
communication = int(input("How good are your communication skills (1-10)? "))

user_input = [[math, science, business, creativity, communication]]
prediction = model.predict(user_input)

print("\n🎓 Based on your answers, we recommend:")
print("👉", prediction[0])

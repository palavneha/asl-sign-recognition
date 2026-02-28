from flask import Flask, request, jsonify
from flask_cors import CORS

from preprocess import preprocess_image
from predictor import predict
from chatbot import get_bot_response

app = Flask(__name__)
CORS(app)


# =========================
# TEST ROUTE (optional but useful)
# =========================
@app.route("/")
def home():
    return "Backend running ✅"


# =========================
# PREDICT ROUTE
# =========================
@app.route("/predict", methods=["POST"])
def predict_api():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    img = preprocess_image(file.read())
    label, confidence = predict(img)

    THRESHOLD = 0.5

    if confidence < THRESHOLD:
        label = "Unknown"
        reply = "Sorry, I couldn't recognize the sign clearly."
    else:
        reply = get_bot_response(label)

    return jsonify({
        "prediction": str(label),
        "confidence": round(confidence, 3),
        "chat_reply": reply
    })




# =========================
# CHATBOT ROUTE  (🔥 missing before)
# =========================
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    message = data.get("message", "")

    reply = get_bot_response(message)

    return jsonify({"reply": reply})


# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    app.run(debug=True)

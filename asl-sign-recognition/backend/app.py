from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from preprocess import preprocess_image
from predictor import predict
from chatbot import get_bot_response
import anthropic
import os

app = Flask(__name__)
CORS(app)

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

from deep_translator import GoogleTranslator
from gtts import gTTS
import io

@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json()
    text = data.get("text", "").strip()
    lang = data.get("lang", "fr")

    if not text:
        return jsonify({"translated": ""})

    try:
        translated = GoogleTranslator(source="en", target=lang).translate(text)
        return jsonify({"translated": translated})
    except Exception as e:
        print(f"Translation error: {e}")
        return jsonify({"error": str(e), "translated": ""}), 500


@app.route("/tts", methods=["POST"])
def tts():
    data = request.get_json()
    text = data.get("text", "").strip()
    lang = data.get("lang", "en")

    if not text:
        return "", 400

    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return app.response_class(
            audio_buffer.read(),
            mimetype="audio/mpeg"
        )
    except Exception as e:
        print(f"TTS error: {e}")
        return jsonify({"error": str(e)}), 500
@app.route("/")
def home():
    return "Backend running ✅"


@app.route("/app")
def serve_frontend():
    frontend_path = os.path.join(os.path.dirname(__file__), '..', 'frontend')
    return send_from_directory(frontend_path, 'index.html')

@app.route("/<path:filename>")
def serve_static(filename):
    frontend_path = os.path.join(os.path.dirname(__file__), '..', 'frontend')
    return send_from_directory(frontend_path, filename)

@app.route("/predict", methods=["POST"])
def predict_api():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img_bytes = file.read()
    img = preprocess_image(img_bytes)

    if img is None:
        return jsonify({
            "prediction": "Unknown",
            "confidence": 0.0,
            "chat_reply": "No hand detected. Please make sure your hand is visible in the frame.",
        })

    label, confidence = predict(img)
    THRESHOLD = 0.80

    if confidence < THRESHOLD:
        label = "Unknown"
        reply = "Sorry, I couldn't recognize the sign clearly."
    else:
        reply = get_bot_response(label)

    return jsonify({
        "prediction": str(label),
        "confidence": round(confidence, 3),
        "chat_reply": reply,
    })


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    message = data.get("message", "")
    reply = get_bot_response(message)
    return jsonify({"reply": reply})


from spellchecker import SpellChecker

spell = SpellChecker()

@app.route("/autocorrect", methods=["POST"])
def autocorrect():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"corrected": text})

    text_upper = text.upper()
    words = text_upper.split()
    corrected_words = []

    for word in words:
        lower = word.lower()
        candidates = spell.candidates(lower)
        if candidates:
            # Pick the closest candidate by edit distance
            best = spell.correction(lower)
            corrected_words.append(best.upper() if best else word)
        else:
            corrected_words.append(word)

    corrected = " ".join(corrected_words)
    return jsonify({"corrected": corrected})

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
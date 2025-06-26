import os
import sys
import logging
import threading
import queue
import time
import traceback
import nltk
import speech_recognition as sr
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from transformers import pipeline
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Optional Gemini API
try:
    import google.generativeai as genai
except ImportError:
    genai = None
    print("Google Gemini module not available.")

# Advanced Emotion Detector
try:
    from emotion_detector import AdvancedEmotionDetector
except ImportError as e:
    AdvancedEmotionDetector = None
    logging.error(f"Import Error: {e}")
    logging.error(f"Python Path: {sys.path}")

# Logging config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK models
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
except Exception as e:
    logger.warning(f"NLTK download error: {e}")

# Flask app init
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'))
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# ----------------- Gemini Chatbot -----------------
class EmotionAwareGeminiChatbot:
    def __init__(self, api_key):
        if genai:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(
                model_name="gemini-1.5-pro",
                generation_config={
                    "temperature": 0.8,
                    "top_p": 0.92,
                    "top_k": 40,
                    "max_output_tokens": 1024
                }
            )
            self.chat = self.model.start_chat(history=[])
        self.emotion_pipeline = pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion")
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.emotion_confidence_threshold = 0.4
        self.emotion_history = []

    def analyze_emotions(self, text):
        results = self.emotion_pipeline(text)
        transformer_emotions = {res["label"]: res["score"] for res in results}
        vader_scores = self.sentiment_analyzer.polarity_scores(text)
        if vader_scores['compound'] >= 0.5:
            transformer_emotions['joy'] = max(transformer_emotions.get('joy', 0), 0.7 * vader_scores['pos'])
        elif vader_scores['compound'] <= -0.5:
            transformer_emotions['sadness'] = max(transformer_emotions.get('sadness', 0), 0.7 * vader_scores['neg'])
            transformer_emotions['anger'] = max(transformer_emotions.get('anger', 0), 0.3 * vader_scores['neg'])
        dominant_emotion = max(transformer_emotions, key=transformer_emotions.get, default="neutral")
        dominant_score = transformer_emotions.get(dominant_emotion, 0.5)
        if dominant_score < self.emotion_confidence_threshold:
            dominant_emotion = "neutral"
            dominant_score = 0.8
        self.emotion_history.append(dominant_emotion)
        return transformer_emotions, dominant_emotion, dominant_score

    def generate_response(self, user_input, emotions, dominant_emotion, confidence_score):
        if not genai:
            return "Gemini API not configured."
        prompt = (
            f"The user's message expresses {dominant_emotion.upper()} (confidence: {confidence_score:.2f}).\n"
        )
        for emotion, score in emotions.items():
            if score > 0.1:
                prompt += f"- {emotion}: {score:.2f}\n"
        prompt += f"\nRespond empathetically to: {user_input}"
        try:
            response = self.chat.send_message(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini response error: {e}")
            return "Sorry, I couldn't generate a response right now."

# ----------------- Voice Emotion Detector -----------------
class EmotionDetector:
    def __init__(self, socketio):
        self.emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.socketio = socketio
        self.stop_event = threading.Event()
        self.audio_queue = queue.Queue()

    def detect_emotions(self, text):
        results = self.emotion_pipeline(text)[0]
        return dict(sorted({res['label']: res['score'] for res in results}.items(), key=lambda x: x[1], reverse=True)[:5])

    def speech_recognition_thread(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            while not self.stop_event.is_set():
                try:
                    audio = self.recognizer.listen(source, phrase_time_limit=3)
                    self.audio_queue.put(audio)
                except:
                    time.sleep(1)

    def audio_processing_thread(self):
        while not self.stop_event.is_set():
            try:
                audio = self.audio_queue.get(timeout=1)
                text = self.recognizer.recognize_google(audio, language='en-US')
                emotions = self.detect_emotions(text)
                self.socketio.emit('emotion_update', {'text': text, 'emotions': emotions})
            except:
                continue

    def start(self):
        self.stop_event.clear()
        threading.Thread(target=self.speech_recognition_thread, daemon=True).start()
        threading.Thread(target=self.audio_processing_thread, daemon=True).start()

    def stop(self):
        self.stop_event.set()

# ----------------- Instances -----------------
GEMINI_API_KEY = 'AIzaSyCwT8adNatXy_xYWdHgBrWtYyk252c4h94'
chatbot = EmotionAwareGeminiChatbot(api_key=GEMINI_API_KEY)
voice_emotion_detector = None
advanced_emotion_detector = AdvancedEmotionDetector() if AdvancedEmotionDetector else None

# ----------------- Routes -----------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/text')
def text_page():
    return render_template('index-text.html')

@app.route('/voice')
def voice_page():
    return render_template('index-voice.html')

@app.route('/real-text')
def real_text_page():
    return render_template('index-real-text.html')  # Reuse same template

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json['message']
        emotions, dominant, confidence = chatbot.analyze_emotions(user_message)
        reply = chatbot.generate_response(user_message, emotions, dominant, confidence)
        return jsonify({'response': reply, 'emotion': dominant, 'emotion_details': {k: round(v, 2) for k, v in emotions.items() if v > 0.1}})
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({'response': 'Error occurred.', 'emotion': 'error'}), 500

@app.route('/analyze', methods=['POST'])
def analyze_emotions():
    if not advanced_emotion_detector:
        return jsonify({'error': 'Advanced detector not initialized', 'status': 'error'}), 500
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        if not text:
            return jsonify({'error': 'Empty input', 'status': 'error'}), 400
        result = advanced_emotion_detector.analyze_multi_sentence(text, per_sentence=True, threshold=0.05)
        return jsonify({
            'overall_emotions': result['overall_emotions'],
            'sentence_emotions': result['sentence_emotions'],
            'metadata': result['analysis_metadata'],
            'status': 'success'
        })
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

@socketio.on('connect')
def on_connect():
    global voice_emotion_detector
    if voice_emotion_detector is None:
        voice_emotion_detector = EmotionDetector(socketio)
    voice_emotion_detector.start()
    emit('status', {'message': 'Connected and listening'})

@socketio.on('disconnect')
def on_disconnect():
    global voice_emotion_detector
    if voice_emotion_detector:
        voice_emotion_detector.stop()

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Page not found', 'status': 'error'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error', 'status': 'error'}), 500

# ----------------- Run -----------------
if __name__ == '__main__':
    print("Launching Unified Emotion Detection Server...")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)

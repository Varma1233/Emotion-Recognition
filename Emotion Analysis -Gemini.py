import os
import logging
import speech_recognition as sr
import pyttsx3
import threading
import queue
from transformers import pipeline
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import time
import numpy as np

# Download necessary NLTK resources
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)

class EmotionAwareGeminiChatbot:
    def __init__(self):
        # Configure Google Gemini API
        genai.configure(api_key="AIzaSyCwT8adNatXy_xYWdHgBrWtYyk252c4h94")
        
        # Initialize Gemini model with better configuration for empathetic responses
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            generation_config={
                "temperature": 0.8,  # Slightly increased for more creative responses
                "top_p": 0.92,
                "top_k": 40,
                "max_output_tokens": 1024,
            },
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
        )
        
        # Initialize chat session with better initial prompt
        self.chat = self.model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": ["I want you to be an emotion-aware assistant that responds with empathy and understanding."],
                },
                {
                    "role": "model",
                    "parts": [
                        "I'll be your emotion-aware assistant. I'll carefully analyze the emotional tone of your messages "
                        "and respond with empathy and understanding. I'll acknowledge your feelings and provide support "
                        "tailored to your emotional state. Feel free to share whatever is on your mind."
                    ],
                },
            ]
        )
        
        # Speech Recognition Setup with improved parameters
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300  # Lower threshold for better sensitivity
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8  # Shorter pause for more natural conversation
        self.microphone = sr.Microphone()
        
        # Text-to-Speech Engine with improved parameters
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty("rate", 175)  # Slightly faster for more natural speech
        self.tts_engine.setProperty("volume", 0.9)
        
        # Try to set a better voice if available
        voices = self.tts_engine.getProperty('voices')
        for voice in voices:
            # Try to find a female voice for more empathetic sound (based on research)
            if "female" in voice.name.lower():
                self.tts_engine.setProperty('voice', voice.id)
                break
        
        # Enhanced Emotion Analysis using multiple models
        self.emotion_pipeline = pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion")
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Adding a confidence threshold for emotion detection
        self.emotion_confidence_threshold = 0.4
        
        # Track conversation history for context
        self.conversation_history = []
        self.emotion_history = []
        
        # Logging Setup with file output
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, f"chatbot_{time.strftime('%Y%m%d_%H%M%S')}.log"))
        console_handler = logging.StreamHandler()
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s',
            handlers=[file_handler, console_handler]
        )
        
        self.logger = logging.getLogger(__name__)
        
        # Welcome Message
        self.welcome_message = (
            "Hello! I'm your emotion-aware assistant. "
            "I can detect how you're feeling and respond accordingly. "
            "Feel free to speak naturally, and I'll do my best to understand both your words and emotions. "
            "You can say 'exit' whenever you'd like to end our conversation."
        )
    
    def recognize_speech(self):
        """Advanced speech recognition with noise handling and improved timeout."""
        try:
            with self.microphone as source:
                print("Listening...")
                self.logger.info("Adjusting for ambient noise...")
                # Better noise adjustment with longer duration
                self.recognizer.adjust_for_ambient_noise(source, duration=1.5)
                self.logger.info("Listening for speech...")
                
                # Increased timeout for better user experience
                audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=10)
            
            self.logger.info("Processing speech...")
            # Using more accurate Google Web Speech API
            text = self.recognizer.recognize_google(audio).strip()
            self.logger.info(f"Recognized Speech: {text}")
            return text
        except sr.WaitTimeoutError:
            self.logger.warning("No speech detected within timeout.")
            return None
        except sr.UnknownValueError:
            self.logger.warning("Could not understand audio")
            return None
        except sr.RequestError as e:
            self.logger.error(f"Speech recognition service error: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error in speech recognition: {e}")
            return None
    
    def analyze_emotions(self, text):
        """Enhanced emotion analysis using multiple methods for better accuracy."""
        try:
            # 1. Use the transformer-based emotion model
            results = self.emotion_pipeline(text)
            transformer_emotions = {result["label"]: result["score"] for result in results}
            
            # 2. Add VADER sentiment analysis for additional insight
            vader_scores = self.sentiment_analyzer.polarity_scores(text)
            
            # 3. Combine both analysis methods for a more nuanced understanding
            combined_emotions = transformer_emotions.copy()
            
            # Map VADER sentiment to emotion categories
            if vader_scores['compound'] >= 0.5:
                combined_emotions['joy'] = max(combined_emotions.get('joy', 0), 0.7 * vader_scores['pos'])
            elif vader_scores['compound'] <= -0.5:
                combined_emotions['sadness'] = max(combined_emotions.get('sadness', 0), 0.7 * vader_scores['neg'])
                combined_emotions['anger'] = max(combined_emotions.get('anger', 0), 0.3 * vader_scores['neg'])
            
            # Determine dominant emotion with confidence check
            dominant_emotion = max(combined_emotions, key=combined_emotions.get)
            dominant_score = combined_emotions[dominant_emotion]
            
            # Fall back to "neutral" if confidence is low
            if dominant_score < self.emotion_confidence_threshold:
                if vader_scores['compound'] > 0.2:
                    dominant_emotion = "joy"
                    dominant_score = 0.5
                elif vader_scores['compound'] < -0.2:
                    dominant_emotion = "sadness"
                    dominant_score = 0.5
                else:
                    dominant_emotion = "neutral"
                    dominant_score = 0.8
            
            # Context-aware emotion smoothing using history
            if self.emotion_history:
                # Reduce emotional "jumping" by considering recent emotions
                recent_emotions = self.emotion_history[-3:] if len(self.emotion_history) >= 3 else self.emotion_history
                if dominant_emotion not in recent_emotions and dominant_score < 0.7:
                    # If new emotion is significantly different but not very strong, smooth the transition
                    most_common_recent = max(set(recent_emotions), key=recent_emotions.count)
                    if most_common_recent in combined_emotions:
                        # Blend with recent emotion if it's a drastic change
                        if combined_emotions[most_common_recent] > dominant_score * 0.7:
                            dominant_emotion = most_common_recent
                            dominant_score = combined_emotions[most_common_recent]
            
            # Update emotion history
            self.emotion_history.append(dominant_emotion)
            
            # Log detailed emotion analysis
            self.logger.info(f"Transformer emotions: {transformer_emotions}")
            self.logger.info(f"VADER sentiment: {vader_scores}")
            self.logger.info(f"Combined analysis: {combined_emotions}")
            self.logger.info(f"Detected emotion: {dominant_emotion} ({dominant_score:.4f})")
            
            return combined_emotions, dominant_emotion, dominant_score
        except Exception as e:
            self.logger.error(f"Emotion analysis error: {e}")
            return {}, "neutral", 0.5
    
    def generate_gemini_response(self, user_input, emotions, dominant_emotion, confidence_score):
        """Generate contextual response using Gemini with enhanced emotion awareness."""
        try:
            # Add to conversation history
            self.conversation_history.append({"role": "user", "text": user_input, "emotion": dominant_emotion})
            
            # Create a more detailed prompt for better emotional context
            emotion_prompt = (
                f"The user's message appears to express primarily {dominant_emotion.upper()} "
                f"(confidence: {confidence_score:.2f}). Their emotional profile shows:"
            )
            
            # Add detailed emotion breakdown
            for emotion, score in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
                if score > 0.1:  # Only include significant emotions
                    emotion_prompt += f"\n- {emotion}: {score:.2f}"
            
            # Add conversation context
            if len(self.conversation_history) > 1:
                emotion_prompt += "\n\nRecent conversation context:"
                context_window = self.conversation_history[-3:] if len(self.conversation_history) > 3 else self.conversation_history
                for i, entry in enumerate(context_window):
                    if i < len(context_window) - 1:  # Skip the current message
                        emotion_prompt += f"\n- Previous {entry['role']}: {entry['text']} (emotion: {entry['emotion']})"
            
            # More specific guidance for the model
            emotion_prompt += (
                f"\n\nRespond with empathy to the user's message. Address their emotional state appropriately for "
                f"someone feeling {dominant_emotion}. Provide supportive and helpful responses that acknowledge "
                f"their feelings. Keep your response concise and natural."
                f"\n\nUser's message: {user_input}"
            )
            
            # Send to Gemini with emotional context
            response = self.chat.send_message(emotion_prompt)
            
            # Clean up the response if needed
            response_text = response.text.strip()
            
            # Store in conversation history
            self.conversation_history.append({"role": "assistant", "text": response_text, "emotion": "supportive"})
            
            return response_text
        except Exception as e:
            self.logger.error(f"Gemini response generation error: {e}")
            
            # Enhanced fallback responses
            fallback_responses = {
                "joy": [
                    "That sounds wonderful! I'm glad you're feeling positive. Would you like to share more about what's bringing you joy?",
                    "It's great to hear you're in good spirits! Those positive feelings are valuable - what's contributing to them?"
                ],
                "sadness": [
                    "I hear that you're feeling down. It's okay to feel this way, and I'm here to listen if you want to talk more.",
                    "I'm sorry you're feeling sad. Sometimes expressing these feelings can help. Would you like to share what's on your mind?"
                ],
                "anger": [
                    "I can sense your frustration. Taking a deep breath might help in the moment. Would you like to talk about what's bothering you?",
                    "I understand you're feeling upset. Your feelings are valid, and I'm here to listen if you want to discuss what's happening."
                ],
                "fear": [
                    "It sounds like you might be worried or anxious. Remember that acknowledging these feelings is an important first step.",
                    "I hear that you're feeling uncertain or afraid. Would talking through what's concerning you help?"
                ],
                "love": [
                    "Those warm feelings sound wonderful to experience. Would you like to share more about this connection?",
                    "It's beautiful to hear about these positive feelings. Would you like to tell me more about what's inspiring them?"
                ],
                "surprise": [
                    "That does sound unexpected! How do you feel about this surprise development?",
                    "Unexpected things can certainly catch us off guard. How are you processing this surprise?"
                ],
                "neutral": [
                    "Thank you for sharing that with me. Would you like to explore this topic further?",
                    "I appreciate you telling me about this. Is there anything specific you'd like to discuss?"
                ]
            }
            
            # Choose a random response from the appropriate category for variety
            responses = fallback_responses.get(dominant_emotion, fallback_responses["neutral"])
            return np.random.choice(responses)
    
    def speak_response(self, response):
        """Enhanced text-to-speech output with error handling."""
        try:
            self.logger.info(f"Response: {response}")
            print(f"\nBot: {response}")
            
            # Break very long responses into chunks for better TTS performance
            if len(response) > 500:
                sentences = response.split('. ')
                chunks = []
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) < 500:
                        current_chunk += sentence + '. '
                    else:
                        chunks.append(current_chunk)
                        current_chunk = sentence + '. '
                
                if current_chunk:
                    chunks.append(current_chunk)
                
                for chunk in chunks:
                    self.tts_engine.say(chunk)
                    self.tts_engine.runAndWait()
            else:
                self.tts_engine.say(response)
                self.tts_engine.runAndWait()
        except Exception as e:
            self.logger.error(f"Speech output error: {e}")
            print(f"Bot (text only due to speech error): {response}")
    
    def run(self):
        """Main chatbot interaction loop with improved session management."""
        print("\n" + "="*50)
        print("EMOTION-AWARE GEMINI CHATBOT")
        print("="*50)
        
        # Speak welcome message
        self.speak_response(self.welcome_message)
        
        print("\nSay 'exit', 'quit', or 'goodbye' to end the session.\n")
        
        conversation_active = True
        consecutive_errors = 0
        
        try:
            while conversation_active:
                # Get user input through speech recognition
                user_input = self.recognize_speech()
                
                # Handle recognition failures
                if not user_input:
                    consecutive_errors += 1
                    if consecutive_errors >= 3:
                        self.speak_response("I'm having trouble hearing you clearly. Please check your microphone or try again in a quieter environment.")
                        consecutive_errors = 0
                    else:
                        self.speak_response("I didn't catch that. Could you please repeat?")
                    continue
                
                consecutive_errors = 0
                print(f"\nYou: {user_input}")
                
                # Check for exit commands
                if user_input.lower() in ["exit", "quit", "goodbye", "bye", "end"]:
                    farewell_message = "Thank you for chatting with me. Take care, and goodbye!"
                    self.speak_response(farewell_message)
                    conversation_active = False
                    break
                
                # Analyze emotions with improved method
                emotions, dominant_emotion, confidence_score = self.analyze_emotions(user_input)
                
                # Display emotion analysis for transparency
                emotion_str = ", ".join([f"{e}: {s:.2f}" for e, s in 
                                       sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]])
                print(f"Detected emotion: {dominant_emotion} ({confidence_score:.2f}) - [{emotion_str}]")
                
                # Generate response with Gemini and emotion awareness
                response = self.generate_gemini_response(user_input, emotions, dominant_emotion, confidence_score)
                
                # Speak response
                self.speak_response(response)
        
        except KeyboardInterrupt:
            print("\n\nChatbot stopped by user.")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            print(f"\n\nAn unexpected error occurred: {e}")
        finally:
            print("\n" + "="*50)
            print("Session ended. Thank you for using the Emotion-Aware Chatbot!")
            print("="*50 + "\n")
            self.logger.info("Chatbot session ended.")

def main():
    try:
        # Create and run the chatbot
        print("Initializing Emotion-Aware Gemini Chatbot...")
        chatbot = EmotionAwareGeminiChatbot()
        chatbot.run()
    except Exception as e:
        print(f"Error starting chatbot: {e}")
        logging.error(f"Critical error: {e}")
        
        # Give users actionable information in case of failure
        print("\nTroubleshooting tips:")
        print("1. Ensure your microphone is properly connected")
        print("2. Check your internet connection")
        print("3. Verify that required Python packages are installed:")
        print("   pip install speechrecognition pyttsx3 transformers nltk google-generativeai numpy")
        print("4. Make sure you have a valid Google API key\n")

if __name__ == "__main__":
    main()
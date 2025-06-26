import torch
import nltk
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F


class AdvancedEmotionDetector:
    def __init__(self, model_name="bhadresh-savani/bert-base-uncased-emotion"):
        """
        Initialize the emotion detection model with proper emotion labels.
        """
        # Ensure NLTK data is downloaded
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            print("Downloading NLTK punkt and punkt_tab data...")
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        # These are the actual emotions that the model is trained on
        self.emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

    def preprocess_text(self, text):
        """
        Preprocess the input text with robust error handling.
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        text = text.strip()
        if not text:
            raise ValueError("Input text cannot be empty")

        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            return inputs
        except Exception as e:
            raise RuntimeError(f"Error during text preprocessing: {str(e)}")

    def analyze_emotions(self, text, threshold=0.05):
        """
        Analyze emotions with improved accuracy reporting.
        """
        try:
            inputs = self.preprocess_text(text)

            with torch.no_grad():
                outputs = self.model(**inputs)
                # Apply softmax to get probabilities
                probabilities = F.softmax(outputs.logits, dim=1)

            emotion_results = []

            # Get raw logits for confidence calculation
            logits = outputs.logits[0]
            # Calculate max logit for scaling
            max_logit = torch.max(logits)

            for i, prob in enumerate(probabilities[0]):
                confidence = prob.item()
                # Calculate normalized confidence score
                normalized_confidence = (logits[i] / max_logit).item()

                if confidence >= threshold:
                    emotion_results.append({
                        'emotion': self.emotion_labels[i],
                        'probability': confidence,
                        'confidence_score': normalized_confidence,
                        'raw_logit': logits[i].item()
                    })

            # Sort by probability
            emotion_results.sort(key=lambda x: x['probability'], reverse=True)
            return emotion_results

        except Exception as e:
            raise RuntimeError(f"Error during emotion analysis: {str(e)}")

    def analyze_multi_sentence(self, text, per_sentence=False, threshold=0.05):
        """
        Analyze emotions across multiple sentences with improved error handling.
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")

        try:
            if per_sentence:
                sentences = nltk.sent_tokenize(text)
            else:
                sentences = [text]

            results = {
                'overall_emotions': self.analyze_emotions(text, threshold),
                'sentence_emotions': [],
                'analysis_metadata': {
                    'num_sentences': len(sentences),
                    'average_sentence_length': sum(len(s.split()) for s in sentences) / len(sentences),
                    'threshold_used': threshold
                }
            }

            if per_sentence:
                for sentence in sentences:
                    if sentence.strip():
                        sentence_emotions = self.analyze_emotions(sentence, threshold)
                        results['sentence_emotions'].append({
                            'sentence': sentence,
                            'emotions': sentence_emotions,
                            'word_count': len(sentence.split())
                        })

            return results

        except Exception as e:
            raise RuntimeError(f"Error during multi-sentence analysis: {str(e)}")


def format_emotion_output(emotion_data):
    """
    Format emotion analysis results for better readability.
    """
    formatted = f"{emotion_data['emotion'].capitalize()}:\n"
    formatted += f"  Probability: {emotion_data['probability']:.4f}\n"
    formatted += f"  Confidence Score: {emotion_data['confidence_score']:.4f}\n"
    formatted += f"  Raw Logit: {emotion_data['raw_logit']:.4f}"
    return formatted


def main():
    """
    Main function with improved error handling and output formatting.
    """
    print("Advanced Real-time Emotion Analysis\n")
    print("Supported emotions:", ", ".join(AdvancedEmotionDetector().emotion_labels))
    print("Type 'exit' to quit.\n")

    try:
        emotion_detector = AdvancedEmotionDetector()
    except Exception as e:
        print(f"Failed to initialize emotion detector: {e}")
        return

    while True:
        try:
            user_input = input("\nEnter text for emotion analysis (or 'exit' to quit): ").strip()

            if user_input.lower() == "exit":
                print("Exiting the emotion analysis.")
                break

            if not user_input:
                print("Please enter some text to analyze.")
                continue

            results = emotion_detector.analyze_multi_sentence(
                user_input,
                per_sentence=True,
                threshold=0.05
            )

            print("\n=== Overall Emotion Analysis ===")
            for emotion in results['overall_emotions']:
                print("\n" + format_emotion_output(emotion))

            if results['sentence_emotions']:
                print("\n=== Sentence-level Analysis ===")
                for i, sentence_data in enumerate(results['sentence_emotions'], 1):
                    print(f"\nSentence {i}: {sentence_data['sentence']}")
                    print(f"Word count: {sentence_data['word_count']}")
                    for emotion in sentence_data['emotions']:
                        print("\n" + format_emotion_output(emotion))

            print("\n=== Analysis Metadata ===")
            meta = results['analysis_metadata']
            print(f"Number of sentences: {meta['num_sentences']}")
            print(f"Average sentence length: {meta['average_sentence_length']:.1f} words")

        except Exception as e:
            print(f"\nError during analysis: {str(e)}")
            print("Please try again with different text.")


if __name__ == "__main__":
    main()
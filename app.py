import gradio as gr
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model
model_name = "joeljoy1912/news-bias-classifier"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Source identifiers to strip at inference time (must match training preprocessing)
SOURCES_TO_STRIP = [
    'vox', 'vox.com', 'vox media',
    'vice', 'vice news', 'vice.com',
    'huffington post', 'huffpost', 'huff post', 'huffingtonpost',
    'buzzfeed', 'buzzfeed news', 'buzzfeednews',
    'guardian', 'the guardian', 'theguardian',
    'new york times', 'nyt', 'nytimes', 'the new york times',
    'reuters', 'associated press', 'ap news', 'ap',
    'bbc', 'bbc news',
    'business insider', 'businessinsider',
    'the hill', 'thehill',
    'npr', 'national public radio',
    'usa today', 'usatoday',
    'fox news', 'foxnews', 'fox',
    'new york post', 'ny post', 'nypost',
    'national review', 'nationalreview',
    'washington times', 'the washington times',
    'breitbart', 'breitbart news',
    'associated press contributed', 'reporting by', 'editing by',
    'compiled by', 'written by',
]

def strip_source_identifiers(text):
    """Remove publication names and outlet-specific boilerplate from article text."""
    cleaned = text
    for source in SOURCES_TO_STRIP:
        pattern = r'\b' + re.escape(source) + r'\b'
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r' {2,}', ' ', cleaned).strip()
    return cleaned

def predict_bias(text):
    """Predict political bias of article"""
    if not text or len(text.strip()) < 100:
        return {"Left": 0.33, "Center": 0.34, "Right": 0.33}

    # Strip source identifiers to match training preprocessing
    text = strip_source_identifiers(text)

    # Tokenize and predict
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # Class probabilities
    left_prob = float(probs[0][0])
    center_prob = float(probs[0][1])
    right_prob = float(probs[0][2])

    # Continuous bias score: -1 (left) to +1 (right)
    bias_score = (-1 * left_prob) + (0 * center_prob) + (1 * right_prob)

    # Granular label from continuous score
    if bias_score <= -0.6:
        bias_label = "Left"
    elif bias_score <= -0.2:
        bias_label = "Left-Leaning"
    elif bias_score <= 0.2:
        bias_label = "Center"
    elif bias_score <= 0.6:
        bias_label = "Right-Leaning"
    else:
        bias_label = "Right"

    results = {
        "Left": left_prob,
        "Center": center_prob,
        "Right": right_prob,
    }

    return results

# Create Gradio interface
demo = gr.Interface(
    fn=predict_bias,
    inputs=gr.Textbox(
        lines=10,
        placeholder="Paste news article here...",
        label="Article Text"
    ),
    outputs=gr.Label(num_top_classes=3, label="Predicted Bias"),
    title="📰 News Bias Classifier",
    description="Analyze political bias in news articles using AI. Produces a continuous bias score from -1 (Left) to +1 (Right). Trained on 37,000+ individually labeled articles.",
    examples=[
        ["The radical left continues pushing their socialist agenda while hardworking Americans suffer under the weight of big government policies. Taxes keep rising and the border remains wide open, yet Democrats refuse to acknowledge the damage they have caused to this great nation and its people."],
        ["The Federal Reserve announced Wednesday that it will maintain interest rates at current levels following its two-day policy meeting. The central bank cited ongoing economic uncertainty and mixed signals from labor market data as reasons for holding steady. Analysts had widely expected the decision, with most forecasting no change until later this year."],
        ["Universal healthcare is a fundamental human right that every civilized nation should guarantee to its citizens. The growing inequality in our healthcare system disproportionately impacts communities of color and working families who cannot afford the skyrocketing costs of prescription drugs and medical treatment in America."],
    ]
)

if __name__ == "__main__":
    demo.launch()
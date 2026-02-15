import gradio as gr
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from newspaper import Article as NewsArticle

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
    """Predict political bias of article text."""
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

    results = {
        "Left": left_prob,
        "Center": center_prob,
        "Right": right_prob,
    }

    return results

def extract_and_predict(url):
    """Fetch article from URL, extract text, and predict bias."""
    if not url or not url.strip():
        return "Error: Please enter a URL.", "", {"Left": 0.33, "Center": 0.34, "Right": 0.33}

    url = url.strip()

    # Basic URL validation
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url

    try:
        article = NewsArticle(url)
        article.download()
        article.parse()
        text = article.text
        title = article.title or "Unknown Title"
    except Exception as e:
        return f"Error: Could not fetch the article. ({str(e)[:100]})", "", {"Left": 0.33, "Center": 0.34, "Right": 0.33}

    if not text or len(text.strip()) < 100:
        return "Error: No article content found at this URL.", "", {"Left": 0.33, "Center": 0.34, "Right": 0.33}

    # Run prediction on extracted text
    result = predict_bias(text)

    # Truncate text for display (keep first 3000 chars)
    display_text = text[:3000]
    if len(text) > 3000:
        display_text += "..."

    return title, display_text, result


# Build Gradio app with gr.Blocks to support multiple endpoints
with gr.Blocks(title="News Bias Classifier") as demo:
    gr.Markdown("# 📰 News Bias Classifier")
    gr.Markdown("Analyze political bias in news articles using AI. Produces a continuous bias score from -1 (Left) to +1 (Right). Trained on 37,000+ individually labeled articles.")

    with gr.Tab("Paste Text"):
        text_input = gr.Textbox(
            lines=10,
            placeholder="Paste news article here...",
            label="Article Text"
        )
        text_btn = gr.Button("Analyze Bias", variant="primary")
        text_output = gr.Label(num_top_classes=3, label="Predicted Bias")
        text_btn.click(
            fn=predict_bias,
            inputs=text_input,
            outputs=text_output,
            api_name="predict_bias"
        )

        gr.Examples(
            examples=[
                ["The radical left continues pushing their socialist agenda while hardworking Americans suffer under the weight of big government policies. Taxes keep rising and the border remains wide open, yet Democrats refuse to acknowledge the damage they have caused to this great nation and its people."],
                ["The Federal Reserve announced Wednesday that it will maintain interest rates at current levels following its two-day policy meeting. The central bank cited ongoing economic uncertainty and mixed signals from labor market data as reasons for holding steady. Analysts had widely expected the decision, with most forecasting no change until later this year."],
                ["Universal healthcare is a fundamental human right that every civilized nation should guarantee to its citizens. The growing inequality in our healthcare system disproportionately impacts communities of color and working families who cannot afford the skyrocketing costs of prescription drugs and medical treatment in America."],
            ],
            inputs=text_input,
        )

    with gr.Tab("From URL"):
        url_input = gr.Textbox(
            lines=1,
            placeholder="Paste article URL here (e.g. https://www.reuters.com/...)",
            label="Article URL"
        )
        url_btn = gr.Button("Extract & Analyze", variant="primary")
        url_title_output = gr.Textbox(label="Article Title", interactive=False)
        url_text_output = gr.Textbox(label="Extracted Text", lines=5, interactive=False, max_lines=10)
        url_bias_output = gr.Label(num_top_classes=3, label="Predicted Bias")
        url_btn.click(
            fn=extract_and_predict,
            inputs=url_input,
            outputs=[url_title_output, url_text_output, url_bias_output],
            api_name="extract_and_predict"
        )

if __name__ == "__main__":
    demo.launch()

# src/app.py
import streamlit as st
import pandas as pd
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from transformers import pipeline
from collections import Counter
from html import escape

# -------------------- Helper Functions --------------------
def extract_comments_from_file(uploaded_file):
    comments = []
    if uploaded_file.name.endswith(".txt"):
        content = uploaded_file.read().decode("utf-8")
        comments = content.splitlines()
    elif uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        comment_cols = [c for c in df.columns if "comment" in c.lower()]
        if comment_cols:
            comments = df[comment_cols[0]].dropna().tolist()
    return comments

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -------------------- Sentiment Analyzer --------------------
@st.cache_resource
def get_sentiment_pipeline():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1
    )

class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = get_sentiment_pipeline()

    def get_overall_sentiment(self, texts, batch_size=64):
        labels, scores = [], []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            res_list = self.analyzer(batch)
            for r in res_list:
                labels.append(r['label'])
                scores.append(r['score'])
        overall_label = max(set(labels), key=labels.count)
        average_score = sum(scores) / len(scores) if scores else 0
        return {"label": overall_label, "average_score": average_score}

# -------------------- Colorful Bullet Summarizer --------------------
class ColorBulletSummarizer:
    def __init__(self, top_keywords=5, max_bullets=5):
        self.top_keywords = top_keywords
        self.max_bullets = max_bullets
        self.colors = ["#e63946", "#457b9d", "#f1faee", "#2a9d8f", "#f4a261"]  # color palette

    def generate_summary(self, comments):
        if not comments:
            return "No comments to summarize."

        text = " ".join([c.strip() for c in comments if c.strip()])
        # Limit text to ~300 words
        words = text.split()
        text = " ".join(words[:300])

        sentences = re.split(r'(?<=[.!?]) +', text)
        words_lower = [w.lower() for w in re.findall(r'\w+', text)]
        stopwords = set([
            "the","and","to","of","in","a","for","on","with","is","this","that","are",
            "as","be","by","an","it","from","or","we","our","can","will","may","should"
        ])
        keywords = [w for w in words_lower if w not in stopwords]
        top_words = [word for word, _ in Counter(keywords).most_common(self.top_keywords)]

        bullets = []
        seen = set()
        for s in sentences:
            clean_s = s.strip()
            if clean_s and clean_s.lower() not in seen:
                seen.add(clean_s.lower())
                # Highlight keywords with colors
                for idx, kw in enumerate(top_words):
                    color = self.colors[idx % len(self.colors)]
                    clean_s = re.sub(
                        fr'\b({escape(kw)})\b',
                        fr'<span style="color:{color};font-weight:bold">\1</span>',
                        clean_s,
                        flags=re.IGNORECASE
                    )
                bullets.append(f"- {clean_s}")
        return "<br>".join(bullets[:self.max_bullets])

# -------------------- Word Cloud Generator --------------------
class WordCloudGenerator:
    def generate_wordcloud(self, texts, max_comments=200):
        text = " ".join(texts[:max_comments])
        if not text.strip():
            return None
        wc = WordCloud(width=800, height=400, background_color="white").generate(text)
        fig, ax = plt.subplots(figsize=(10,5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        return fig

# -------------------- Streamlit Page --------------------
st.set_page_config(page_title="Sentiment Analysis App", layout="centered")
st.title("💬 e-Consultation Comments Analyzer")
st.write(
    "Enter a comment OR upload a CSV/TXT file to analyze sentiment, generate a short colored bullet-point summary, and create a word cloud."
)

comment_input = st.text_area("📝 Enter your comment here:")
uploaded_file = st.file_uploader("📂 Or upload a file (.txt or .csv)", type=["txt", "csv"])

if st.button("🔍 Analyze"):
    comments = []
    if uploaded_file:
        comments = extract_comments_from_file(uploaded_file)
    elif comment_input.strip():
        comments = [comment_input.strip()]
    else:
        st.warning("⚠️ Please enter a comment or upload a file.")

    if comments:
        cleaned_comments = [clean_text(c) for c in comments if c.strip()]

        with st.spinner("Analyzing comments..."):
            # Sentiment
            sentiment_analyzer = SentimentAnalyzer()
            overall_sentiment = sentiment_analyzer.get_overall_sentiment(cleaned_comments)

            st.subheader("📊 Overall Sentiment")
            st.write(f"**Label:** {overall_sentiment['label']}")
            st.write(f"**Average Confidence:** {overall_sentiment['average_score']:.2f}")

            # Summary
            summarizer = ColorBulletSummarizer()
            summary_html = summarizer.generate_summary(cleaned_comments)
            st.subheader("Summary of Comments")
            st.markdown(summary_html, unsafe_allow_html=True)

            # Word Cloud
            wordcloud_gen = WordCloudGenerator()
            fig = wordcloud_gen.generate_wordcloud(cleaned_comments)
            if fig:
                st.subheader("☁️ Word Cloud")
                st.pyplot(fig)
            else:
                st.write("No text available to generate a word cloud.")

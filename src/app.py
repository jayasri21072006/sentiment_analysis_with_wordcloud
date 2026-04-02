# src/app.py
import csv
import re
from collections import Counter
from html import escape
from io import BytesIO

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from transformers import pipeline
from wordcloud import WordCloud

MAX_COMMENTS = 500
MAX_TEXT_CHARS = 50000
COMMENT_TOKENS = ("comment", "review", "feedback", "text", "message")


def extract_comments_from_file(uploaded_file):
    if uploaded_file.name.endswith(".txt"):
        uploaded_file.seek(0)
        content = uploaded_file.read().decode("utf-8", errors="ignore")
        return [line.strip() for line in content.splitlines() if line.strip()][:MAX_COMMENTS]

    if not uploaded_file.name.endswith(".csv"):
        return []

    uploaded_file.seek(0)
    raw_bytes = uploaded_file.read()
    sample_text = raw_bytes[:4096].decode("utf-8", errors="ignore")

    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=",;\t|")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ","

    dataframe = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            dataframe = pd.read_csv(
                BytesIO(raw_bytes),
                sep=delimiter,
                encoding=encoding,
                engine="c",
                on_bad_lines="skip",
                nrows=MAX_COMMENTS,
            )
            if not dataframe.empty:
                break
        except Exception:
            dataframe = None

    if dataframe is None or dataframe.empty:
        return []

    comment_cols = [
        column
        for column in dataframe.columns
        if any(token in str(column).lower() for token in COMMENT_TOKENS)
    ]

    if comment_cols:
        source_col = comment_cols[0]
    else:
        text_like_cols = [
            column
            for column in dataframe.columns
            if dataframe[column]
            .dropna()
            .map(lambda value: isinstance(value, str) and value.strip() != "")
            .any()
        ]
        if not text_like_cols:
            return []
        source_col = text_like_cols[0]

    comments = dataframe[source_col].dropna().astype(str).str.strip().tolist()
    return [comment for comment in comments if comment][:MAX_COMMENTS]


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource
def get_sentiment_pipeline():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1,
    )


class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = get_sentiment_pipeline()

    def get_overall_sentiment(self, texts, batch_size=64):
        labels = []
        scores = []

        for index in range(0, len(texts), batch_size):
            batch = texts[index:index + batch_size]
            results = self.analyzer(batch)
            for result in results:
                labels.append(result["label"])
                scores.append(result["score"])

        overall_label = max(set(labels), key=labels.count) if labels else "NEUTRAL"
        average_score = sum(scores) / len(scores) if scores else 0.0
        return {"label": overall_label, "average_score": average_score}


class ColorBulletSummarizer:
    def __init__(self, top_keywords=5, max_bullets=5):
        self.top_keywords = top_keywords
        self.max_bullets = max_bullets
        self.colors = ["#e63946", "#457b9d", "#f1faee", "#2a9d8f", "#f4a261"]

    def generate_summary(self, comments):
        if not comments:
            return "No comments to summarize."

        text = " ".join(comment.strip() for comment in comments if comment.strip())[:MAX_TEXT_CHARS]
        words = text.split()
        text = " ".join(words[:300])

        sentences = re.split(r"(?<=[.!?]) +", text)
        words_lower = [word.lower() for word in re.findall(r"\w+", text)]
        stopwords = {
            "the", "and", "to", "of", "in", "a", "for", "on", "with", "is", "this", "that", "are",
            "as", "be", "by", "an", "it", "from", "or", "we", "our", "can", "will", "may", "should",
        }
        keywords = [word for word in words_lower if word not in stopwords]
        top_words = [word for word, _ in Counter(keywords).most_common(self.top_keywords)]

        bullets = []
        seen = set()
        for sentence in sentences:
            clean_sentence = sentence.strip()
            if clean_sentence and clean_sentence.lower() not in seen:
                seen.add(clean_sentence.lower())
                for idx, keyword in enumerate(top_words):
                    color = self.colors[idx % len(self.colors)]
                    clean_sentence = re.sub(
                        rf"\b({escape(keyword)})\b",
                        rf'<span style="color:{color};font-weight:bold">\1</span>',
                        clean_sentence,
                        flags=re.IGNORECASE,
                    )
                bullets.append(f"- {clean_sentence}")

        return "<br>".join(bullets[:self.max_bullets])


class WordCloudGenerator:
    def generate_wordcloud(self, texts, max_comments=200):
        text = " ".join(texts[:max_comments])
        if not text.strip():
            return None

        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        return fig


st.set_page_config(page_title="Sentiment Analysis App", layout="centered")
st.title("e-Consultation Comments Analyzer")
st.write(
    "Enter a comment or upload a CSV/TXT file to analyze sentiment, generate a short summary, and create a word cloud."
)

comment_input = st.text_area("Enter your comment here:")
uploaded_file = st.file_uploader("Or upload a file (.txt or .csv)", type=["txt", "csv"])

if st.button("Analyze"):
    try:
        if uploaded_file:
            comments = extract_comments_from_file(uploaded_file)
        elif comment_input.strip():
            comments = [comment_input.strip()]
        else:
            comments = []
            st.warning("Please enter a comment or upload a file.")

        if comments:
            cleaned_comments = [clean_text(comment) for comment in comments if comment.strip()]
            if not cleaned_comments:
                st.warning("No usable text was found in the input.")
            else:
                with st.spinner("Analyzing comments..."):
                    sentiment_analyzer = SentimentAnalyzer()
                    overall_sentiment = sentiment_analyzer.get_overall_sentiment(cleaned_comments)

                    st.subheader("Overall Sentiment")
                    st.write(f"**Label:** {overall_sentiment['label']}")
                    st.write(f"**Average Confidence:** {overall_sentiment['average_score']:.2f}")
                    st.caption(
                        f"Processed {len(cleaned_comments)} comment(s), up to {MAX_COMMENTS} from uploaded files."
                    )

                    summarizer = ColorBulletSummarizer()
                    summary_html = summarizer.generate_summary(cleaned_comments)
                    st.subheader("Summary of Comments")
                    st.markdown(summary_html, unsafe_allow_html=True)

                    wordcloud_gen = WordCloudGenerator()
                    fig = wordcloud_gen.generate_wordcloud(cleaned_comments)
                    if fig:
                        st.subheader("Word Cloud")
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.write("No text available to generate a word cloud.")
        elif uploaded_file:
            st.warning("No comments could be extracted from that file.")
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")

st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; font-size:14px; color:gray;">
        &copy; 2026 JAYASRI T. All Rights Reserved.<br>
        Licensed under the MIT License.<br>
        Developed for educational and demonstration purposes only.
    </div>
    """,
    unsafe_allow_html=True,
)

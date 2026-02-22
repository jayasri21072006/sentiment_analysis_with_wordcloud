from transformers import pipeline
import re
from collections import Counter

class Summarizer:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        """
        Initializes the transformer summarization pipeline.
        """
        self.summarizer = pipeline("summarization", model=model_name)

    def generate_summary(self, comments, max_length=60, min_length=20, top_keywords=5, max_bullets=5):
        """
        Generates a short, crisp, bullet-point summary from comments.
        Highlights important keywords.

        Returns:
        - str: bullet-pointed summary with highlighted keywords
        """
        if not comments:
            return "No comments to summarize."

        # Combine all comments
        text = " ".join([c.strip() for c in comments if c.strip()])
        if not text:
            return "No valid comments to summarize."

        # Split into chunks to avoid truncation
        max_chunk_size = 1000
        chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]

        # Generate summaries for each chunk
        summaries = []
        for chunk in chunks:
            res = self.summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
            summaries.append(res[0]['summary_text'])

        combined_summary = " ".join(summaries)

        # Extract top keywords
        words = re.findall(r'\w+', combined_summary.lower())
        stopwords = set([
            "the","and","to","of","in","a","for","on","with","is","this","that","are",
            "as","be","by","an","it","from","or","we","our","can","will","may","should"
        ])
        keywords = [w for w in words if w not in stopwords]
        top_words = [word for word, _ in Counter(keywords).most_common(top_keywords)]

        # Split into sentences, remove duplicates, and highlight keywords
        sentences = list(dict.fromkeys(re.split(r'(?<=[.!?]) +', combined_summary)))
        bullets = []
        for s in sentences:
            if s.strip():
                for kw in top_words:
                    s = re.sub(fr'\b({kw})\b', r'**\1**', s, flags=re.IGNORECASE)
                bullets.append(f"- {s.strip()}")

        # Keep only top N bullets for clarity
        return "\n".join(bullets[:max_bullets])

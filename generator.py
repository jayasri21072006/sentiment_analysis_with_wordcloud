from wordcloud import WordCloud
import matplotlib.pyplot as plt

class WordCloudGenerator:
    def __init__(self, width=800, height=400, background_color="white"):
        """
        Initializes the WordCloud generator.
        """
        self.width = width
        self.height = height
        self.background_color = background_color

    def generate(self, text):
        """
        Generates a WordCloud object from the given text.
        Returns None if text is empty.
        """
        if not text.strip():
            return None  # Handle empty text

        wc = WordCloud(
            width=self.width,
            height=self.height,
            background_color=self.background_color
        ).generate(text)
        return wc

    def plot(self, wordcloud):
        """
        Returns a matplotlib figure for Streamlit display.
        """
        if wordcloud is None:
            return None

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        plt.tight_layout(pad=0)
        return fig

    def create_wordcloud_from_comments(self, comments):
        """
        Combines multiple comments and generates a WordCloud.
        """
        text = " ".join([c.strip() for c in comments if c.strip() != ""])
        return self.generate(text)

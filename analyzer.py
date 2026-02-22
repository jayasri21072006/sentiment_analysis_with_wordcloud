class SentimentAnalyzer:
    def __init__(self, model):
        self.model = model

    def analyze_sentiment(self, comment):
        # Here you would implement the logic to analyze the sentiment of the comment
        # For example, using a pre-trained model to predict sentiment
        sentiment_score = self.model.predict(comment)  # Placeholder for actual prediction logic
        return sentiment_score

    def get_overall_sentiment(self, comments):
        total_score = 0
        for comment in comments:
            total_score += self.analyze_sentiment(comment)
        overall_sentiment = total_score / len(comments) if comments else 0
        return overall_sentiment
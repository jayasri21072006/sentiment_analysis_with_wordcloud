# Sentiment Analysis Application

This project is an AI-assisted tool designed to analyze comments received through the eConsultation module of the MCA21 portal. The application aims to facilitate the review of stakeholder feedback on proposed amendments and draft legislations by providing sentiment analysis, summary generation, and visual representation of keywords through a word cloud.

## Features

- **Sentiment Analysis**: Predicts the sentiment of individual comments and calculates the overall sentiment from a collection of comments.
- **Summary Generation**: Produces concise summaries of stakeholder comments to highlight key observations.
- **Word Cloud Generation**: Visualizes the frequency of keywords used in comments, providing insights into common themes and concerns.

## Project Structure

```
sentiment-analysis-app
├── src
│   ├── app.py               # Main entry point of the application
│   ├── sentiment
│   │   └── analyzer.py      # Contains the SentimentAnalyzer class
│   ├── summary
│   │   └── summarizer.py    # Contains the Summarizer class
│   ├── wordcloud_utils
│   │   └── generator.py      # Contains the WordCloudGenerator class
│   └── utils
│       └── helpers.py       # Utility functions for data processing
├── requirements.txt         # Project dependencies
├── README.md                # Project documentation
└── .gitignore               # Files and directories to ignore in version control
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd sentiment-analysis-app
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```
   python src/app.py
   ```

2. Access the application through your web browser at `http://localhost:5000`.

3. Submit comments for sentiment analysis, summary generation, and to view the word cloud.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
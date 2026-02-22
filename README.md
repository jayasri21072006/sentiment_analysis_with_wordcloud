# 📊 Sentiment Analysis Application

An AI-powered web application designed to analyze stakeholder comments received through the MCA21 eConsultation module.

The system performs sentiment analysis, generates concise summaries, and visualizes key themes using a dynamic word cloud — helping decision-makers quickly interpret public feedback on draft legislations and amendments.

---

## 🚀 Features

- **Sentiment Analysis**
  - Classifies comments as Positive, Neutral, or Negative
  - Computes overall sentiment score

- **Automated Summary Generation**
  - Extracts key insights from multiple comments
  - Highlights major themes and concerns

- **Word Cloud Visualization**
  - Displays frequently used keywords
  - Provides quick visual understanding of dominant topics

---

## 🏗️ Project Structure


sentiment-analysis-app
├── src
│ ├── app.py
│ ├── sentiment
│ │ └── analyzer.py
│ ├── summary
│ │ └── summarizer.py
│ ├── wordcloud_utils
│ │ └── generator.py
│ └── utils
│ └── helpers.py
├── requirements.txt
├── README.md
└── .gitignore


---

## 🛠️ Tech Stack

- Python
- Streamlit
- NLP Models
- WordCloud
- Pandas
- Matplotlib

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/jayasri21072006/sentiment_analysis_with_wordcloud.git
cd sentiment-analysis-app

Install dependencies:

pip install -r requirements.txt
▶️ Run the Application
streamlit run src/app.py

Open in browser:

http://localhost:8501
🎯 Use Case

This application helps:

Government bodies

Policy analysts

Legal researchers

Public consultation teams

Quickly analyze large volumes of stakeholder feedback.

🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

📜 License

This project is licensed under the MIT License.

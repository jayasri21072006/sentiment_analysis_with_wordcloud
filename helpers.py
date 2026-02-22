import re
import nltk

# Download NLTK stopwords once
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOP_WORDS = set(stopwords.words('english'))

def clean_text(text):
    """
    Cleans input text by:
    - Removing special characters and digits
    - Converting to lowercase
    - Removing extra whitespace
    """
    if not text:
        return ""
    
    # Remove special characters and digits
    text = re.sub(r'[^A-Za-z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_text(text):
    """
    Tokenizes text into a list of words using nltk.word_tokenize
    """
    if not text:
        return []
    return word_tokenize(text)

def remove_stopwords(tokens):
    """
    Removes common English stopwords from a list of tokens
    """
    if not tokens:
        return []
    return [word for word in tokens if word not in STOP_WORDS]

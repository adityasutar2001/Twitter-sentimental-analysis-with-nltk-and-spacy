import re
import nltk
import spacy
from nltk.corpus import stopwords

# Download once
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    """Clean text using regex, NLTK, and spaCy"""
    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove non-alphabetic characters
    text = re.sub(r"[^a-z\s]", "", text)

    # Tokenize with spaCy
    doc = nlp(text)

    # Lemmatize & remove stopwords
    tokens = [token.lemma_ for token in doc if token.text not in stop_words and not token.is_punct]

    return " ".join(tokens)

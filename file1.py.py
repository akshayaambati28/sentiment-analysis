import tweepy
import pandas as pd
import re
import nltk
import streamlit as st
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Download required NLTK data
nltk.download("vader_lexicon")
nltk.download("stopwords")
nltk.download("punkt")

# 🔑 Twitter API Authentication (Replace with your actual API keys)
api_key = "bCFZ9qVlt87bQAbHvbRlMWHVm"
api_secret = "OpZdYbNbnbdt6GWjbEb9WqvmKhWjvMjXidHPeJGLDUyIosP4Ol"
access_token = "1892636508315910144-9bsOzTFciGdziOpbDTkhnHZEyz3oYh"
access_secret = "qOXIG026xW0i7qwXJSULfd9F3dsVZ0pjUUjNjHUfCwRgj"

# Authenticate Twitter API
auth = tweepy.OAuthHandler(api_key, api_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

# Function to fetch tweets using Tweepy Cursor
def fetch_tweets(query, count=100):
    tweets = tweepy.Cursor(api.search_tweets, q=query, lang="en", tweet_mode="extended").items(count)
    return pd.DataFrame([[tweet.full_text] for tweet in tweets], columns=["Tweet"])

# Function to clean tweet text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+|#\w+", "", text)  # Remove mentions and hashtags
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = re.sub(r"\d+", "", text)  # Remove numbers

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    return " ".join([word for word in tokens if word not in stop_words])

# Function to analyze sentiment using VADER
def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(text)
    if score["compound"] > 0.05:
        return "Positive"
    elif score["compound"] < -0.05:
        return "Negative"
    else:
        return "Neutral"

# Train a Machine Learning model for Sentiment Analysis
def train_ml_model(data):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(data["Cleaned_Tweet"])
    y = data["Sentiment"].map({"Positive": 1, "Neutral": 0, "Negative": -1})  # Map sentiment labels to numbers
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred) * 100
    st.write(f"✅ Model Accuracy: {accuracy:.2f}%")
    return model, vectorizer

# Create Streamlit Dashboard
def create_dashboard(data):
    st.title("📊 Twitter Sentiment Analysis")
    st.subheader("Fetched & Processed Tweets")
    st.write(data.head())

    # Sentiment Distribution Pie Chart
    sentiment_counts = data["Sentiment"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct="%1.1f%%", colors=["green", "gray", "red"])
    st.pyplot(fig)

# Streamlit App Logic
st.sidebar.header("Twitter Sentiment Analysis")
query = st.sidebar.text_input("Enter a keyword/hashtag:", value="AI news")
num_tweets = st.sidebar.slider("Number of tweets to analyze:", 50, 500, 100)

if st.sidebar.button("Fetch & Analyze Tweets"):
    with st.spinner("Fetching Tweets..."):
        data = fetch_tweets(query, count=num_tweets)

    data["Cleaned_Tweet"] = data["Tweet"].apply(clean_text)
    data["Sentiment"] = data["Cleaned_Tweet"].apply(analyze_sentiment)

    model, vectorizer = train_ml_model(data)
    create_dashboard(data)

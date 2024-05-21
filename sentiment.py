import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')


data = pd.read_csv('tweets.csv')

def preprocess(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words and word.isalpha()]
    return " ".join(filtered_text)


data['cleaned_tweets'] = data['content'].apply(preprocess)  


sia = SentimentIntensityAnalyzer()
data['sentiment'] = data['cleaned_tweets'].apply(lambda x: sia.polarity_scores(x)['compound'])


data['sentiment_label'] = data['sentiment'].apply(lambda x: 'positive' if x > 0 else ('neutral' if x == 0 else 'negative'))


print(data[['content', 'cleaned_tweets', 'sentiment', 'sentiment_label']].head())


data.to_csv('tweets_with_sentiment.csv', index=True)



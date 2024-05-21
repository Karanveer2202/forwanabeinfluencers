import streamlit as st
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import plotly.express as px


nltk.download('vader_lexicon')


data = pd.read_csv('tweets.csv')  

sia = SentimentIntensityAnalyzer()

data['sentiment'] = data['content'].apply(lambda x: sia.polarity_scores(x)['compound'])
data['positive_score'] = data['content'].apply(lambda x: sia.polarity_scores(x)['pos'] * 100)
data['neutral_score'] = data['content'].apply(lambda x: sia.polarity_scores(x)['neu'] * 100)
data['negative_score'] = data['content'].apply(lambda x: sia.polarity_scores(x)['neg'] * 100)

data['sentiment_label'] = data['sentiment'].apply(lambda x: 'positive' if x > 0 else ('neutral' if x == 0 else 'negative'))

def get_most_viral_tweets(celebrity, n=5):
    celeb_tweets = data[data['author'] == celebrity]
    if celeb_tweets.empty:
        return pd.DataFrame(columns=['content', 'number_of_shares', 'positive_score', 'neutral_score', 'negative_score']) #this was a vscode suggestion
    viral_tweets = celeb_tweets.nlargest(n, 'number_of_shares')[['content', 'number_of_shares', 'positive_score', 'neutral_score', 'negative_score']]
    return viral_tweets #return a dataframe


# Streamlitapp
st.title('Celebrity Tweets Sentiment Analysis')

st.write('This app analyzes the sentiment of tweets of different celebrities and displays the most viral tweets of a selected celebrity.')
celebrities = data['author'].unique()

selected_celebrity = st.selectbox('Select a Celebrity', celebrities) #why don't they just call it a dropdown?

if selected_celebrity:
    st.subheader(f'Most Viral Tweets of {selected_celebrity}')
    viral_tweets = get_most_viral_tweets(selected_celebrity)
    st.write(viral_tweets)


    if not viral_tweets.empty:
        avg_positive = viral_tweets['positive_score'].mean()#calculated the avg of the scores
        avg_neutral = viral_tweets['neutral_score'].mean()
        avg_negative = viral_tweets['negative_score'].mean()

        sentiments = ['Positive', 'Neutral', 'Negative']
        scores = [avg_positive, avg_neutral, avg_negative]

        fig = px.bar(x=sentiments, y=scores, labels={'x': 'Sentiment', 'y': 'Average Score'},  
                     title=f'Sentiment Analysis of {selected_celebrity}\'s Most Viral Tweets',
                     color=sentiments, color_discrete_map={'Positive':'green', 'Neutral':'blue', 'Negative':'red'})

        st.plotly_chart(fig)
        


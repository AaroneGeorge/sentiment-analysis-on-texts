import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

def SentimentAnalysisModel(input_text):

    df = pd.read_csv('preprocessed_dataset.csv')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')  
    X_tfidf = joblib.load('X_tfidf.pkl') 
    y = df['sentiment']
   
    model = joblib.load('model.pkl')

    

    input_features = tfidf_vectorizer.transform([input_text])

    prediction = model.predict(input_features)


    print(f"Sentiment Analysis Result: {prediction[0]}")


import pyfiglet

result = pyfiglet.figlet_format("Sentiment Analysis Project", font = "slant" )
print(result)
input_text = input("Enter your text: ")
SentimentAnalysisModel(input_text)
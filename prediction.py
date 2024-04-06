## let's start the python code from here
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt 
# import seaborn as sns

from sklearn.model_selection import train_test_split,GridSearchCV
# from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re #to remove the special characters in the text
from sklearn.feature_extraction.text import CountVectorizer
##importing models of the day
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.metrics import accuracy_score
import json
import sys
import joblib 

##read the data here 
# data = pd.read_csv('../text.csv')

## user input will be recived here
# input_text = json.loads(sys.stdin.readline().strip())

content =  "As the shadows lengthen and strange noises echo through the house, every creak and whisper sends shivers down my spine, and I can't help but fear what lurks in the darkness."

# content = input_text['content']
# content = str(content)

# print('The input_text here is ',content)

##preprocessing function
pattern = r'[^a-zA-Z0-9\s]' #anything inside this will be removed from the text
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
##preprocessing function here
def preprocess(text):
    clean_text = re.sub(pattern,'',text) #replace with empty string
    
    # tokenize the given clean text
    tokens = word_tokenize(clean_text)
    
    # remove stop words from the text
    filtered_token = [word for word in tokens if word.lower() not in stop_words]
    
    # apply porter stemmer 
    stemmed_token = [ps.stem(word) for word in filtered_token]
    
    return ' '.join(stemmed_token)

##count vectorizer function here

##the user input we expect to get
preprocessed_text = preprocess(content)
##vectorize the user input
cv = joblib.load('countvectorizer2.pkl')

model = joblib.load('sentiment_analysis2.pkl')

vectorized_text = cv.transform([preprocessed_text])

print(vectorized_text,'\n')

print('The answer is ',model.predict(vectorized_text))
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

##read the data here 
data = pd.read_csv('../text.csv')

## user input will be recived here
input_text = json.loads(sys.stdin.readline().strip())

content = input_text['content']
content = str(content)

print('The input_text here is ',content)

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

def CountVectorize(data):
    cv = CountVectorizer(max_features=5000) #that means we will select the most frequent 5000 words
    cv.fit(data)
    trf = cv.transform(data)
    trf = trf.toarray()
    
    return trf

## first 5000 data because we have too many entries in the dataset
data = data[:50000]

data['text'] = data['text'].apply(preprocess)

trf = CountVectorize(data['text'])

##the user input we expect to get
preprocessed_text = preprocess(content)
##vectorize the user input
vectorized_text = CountVectorize([preprocessed_text])

##check if the vectors shapes are equal
if trf.shape[1] != vectorized_text.shape[1]:
    print("Number of features in user_vector and vectors don't match. Adjusting...")

    cv_adjusted = CountVectorizer(max_features=trf.shape[1], stop_words='english')
    vectors_adjusted = cv_adjusted.fit_transform(data['text']).toarray()

    user_vector = cv_adjusted.transform([preprocessed_text]).toarray()
    

##split the data into train/test split
X = vectors_adjusted
y = data['label']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

print('This is the content at the end',content)
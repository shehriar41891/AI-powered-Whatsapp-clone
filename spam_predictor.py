##importent packages
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re #to remove the special characters in the text
import joblib

## user input
user_input = "U dun say so early hor... U c already then say..."

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

tfidf = joblib.load('spam_tfidf.pkl')

## fit the new user_input
vectorized_input = tfidf.transform([user_input])

spam_predictor = joblib.load('spam_predictor.pkl')

prediction = spam_predictor.predict(vectorized_input)

print('The prediction is ',prediction)
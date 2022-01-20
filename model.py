import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
import re
import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime
import warnings
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,balanced_accuracy_score
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import CountVectorizer


def scrub_words(text):
    """Basic cleaning of texts."""
    
    # remove html markup
    text=re.sub("(<.*?>)"," ",text)
    
    # remove unneccessary words
    text = text.replace("RE:","")
    text = text.replace("FW:","")
    text = text.replace("_"," ")
    
    #remove non-ascii and digits
    text=re.sub("(\\W|\\d)"," ",text)
    
    #remove whitespace
    text = text.strip()
    text = re.sub(' +', ' ',text)
    
    return text

def lemmatize_sentence(sentence):
    #tokenize the sentence and find the POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    #tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            #if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:
            #else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)

def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def loadingdata(path,name):
    Sample_data = pd.read_csv(path)
    df = Sample_data.dropna(subset=['user_sentiment'])
    df['user_sentiment']= df['user_sentiment'].map({'Positive':1 , 'Negative':0})
    df['reviews_text'] = df['reviews_text'].astype('str')
    df['reviews_text'] = df['reviews_text'].str.replace('[^\w\s]','')
    stop = stopwords.words('english')
    df['reviews_text'] = df['reviews_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    df['reviews_text']=df['reviews_text'].apply(lambda x: scrub_words(x))
    df['reviews_text']= df['reviews_text'].str.lower()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    classifier = LogisticRegression(class_weight = "balanced", C=0.5, solver='sag')
    model(df,classifier,"LRModel")
    
    return prediction(name,Sample_data)
        
    
def model_split(df) :
    '''This function splits data to train and test, then vectorized reviews '''
    
    # split train-test
    X_train, X_test, y_train, y_test = train_test_split(df['reviews_text'], 
                                                        df['user_sentiment'], test_size=0.2, random_state=42)
    
    # define vectorize and fit to data     
    word_vectorizer = TfidfVectorizer(sublinear_tf=True,strip_accents='unicode',
        analyzer='word',token_pattern=r'\w{1,}',stop_words='english')

    word_vectorizer.fit(df['reviews_text'])
    
    # train - test vectorized features - tranforming to suitable format for modeling
    train_word_features = word_vectorizer.transform(X_train) 
    test_word_features = word_vectorizer.transform(X_test)

    # handling class imbalance 
    counter = Counter(y_train)
    sm = SMOTE()
    X_train_transformed_sm, y_train_sm = sm.fit_resample(train_word_features, y_train)
    counter = Counter(y_train_sm)
    pickle.dump(word_vectorizer.vocabulary_, open("vector", "wb"))
    return X_train_transformed_sm , test_word_features, y_train_sm, y_test 

def model(df,classifier,filename):
    '''this function gives modeling results and confusion matrix also'''
    train_word_features,test_word_features,y_train,y_test = model_split(df)
    classifier.fit(train_word_features, y_train)
    # calculating results 
    y_pred_train = classifier.predict(train_word_features)
    y_pred = classifier.predict(test_word_features)
    
    #for smart printing (learned from our lead instructor Bryan Arnold)
   
    with open(filename, 'wb') as files:
      pickle.dump(classifier, open(filename, 'wb'))
    # plot confusion matrix
    cm = confusion_matrix(y_test, y_pred) 

@app.route('/login', methods=['GET', 'POST'])   
def prediction(name,reviews):
    print(name)
    pipeline = pickle.load(open('pickle/user_based_recomm.pkl', 'rb'))
    sr = pipeline.loc[name].sort_values(ascending=False)[0:20] ## series
    top_20_products = pd.DataFrame({'name':sr.index})
    top_20_reviews = reviews[reviews['name'].isin(top_20_products['name'])][['name','reviews_text']] 
    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open('pickle/vector', 'rb')))
    test_data_features = transformer.fit_transform(loaded_vec.fit_transform(top_20_reviews['reviews_text']))
    loaded_model = pickle.load(open("pickle/LRModel", 'rb'))
    result1 = loaded_model.predict(test_data_features)
    top_20_reviews['sentiment'] = result1.tolist()
    top = top_20_reviews.groupby(['name']).mean()
    top5 = top.sort_values(by='sentiment',ascending=False)[:5]
    top5.reset_index(level=0, inplace=True)
    top_5_products= top5['name']
    top_5_product = pd.DataFrame({'name':top_5_products})
    return top_5_product  


loadingdata('dataset/sample30.csv',"joshua")
print(loadingdata('dataset/sample30.csv',"joshua"))
#!/usr/bin/env python
# coding: utf-8

# # Replication for results in Davidson et al. 2017. "Automated Hate Speech Detection and the Problem of Offensive Language"

# In[1]:


import pandas as pd
import numpy as np
import pickle
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem.porter import *
import string
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer as VS
from textstat.textstat import *

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC

import matplotlib.pyplot as plt
import seaborn
import preprocessor as p
from nltk.tokenize import TweetTokenizer as tt
from string import punctuation

#from autocorrect import spell
#nltk.download('stopwords')

#get_ipython().run_line_magic('matplotlib', 'inline')


# ## Loading the data

# In[2]:


df = pd.read_csv("../data/labeled_data.csv")
lyrics = [line.rstrip('\n') for line in open("all_tweets_lyrics.txt")]

# In[3]:


df


# In[4]:


df.describe()


# In[5]:


df.columns


# ### Columns key:
# count = number of CrowdFlower users who coded each tweet (min is 3, sometimes more users coded a tweet when judgments were determined to be unreliable by CF).
# 
# 
# hate_speech = number of CF users who judged the tweet to be hate speech.
# 
# 
# offensive_language = number of CF users who judged the tweet to be offensive.
# 
# 
# neither = number of CF users who judged the tweet to be neither offensive nor non-offensive.
# 
# 
# class = class label for majority of CF users.
# 
#     0 - hate speech
#     1 - offensive  language
#     2 - neither
# 
# tweet = raw tweet text
# 

# In[6]:


df['class'].hist()


# This histogram shows the imbalanced nature of the task - most tweets containing "hate" words as defined by Hatebase were 
# only considered to be offensive by the CF coders. More tweets were considered to be neither hate speech nor offensive language than were considered hate speech.

# In[7]:


tweets=df.tweet


# ## Feature generation

# In[8]:


stopwords=stopwords = nltk.corpus.stopwords.words("english")

#stopwords = nltk.corpus.stopwords.words("english")


#nltk.download('stopwords', quiet=True, raise_on_error=True)
#stop_words = set(nltk.corpus.stopwords.words('english'))
#stopwords = nltk.word_tokenize(' '.join(nltk.corpus.stopwords.words('english')))

other_exclusions = ["#ff", "ff", "rt"]
stopwords.extend(other_exclusions)


stemmer = PorterStemmer()


def preprocess(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, '', parsed_text)
    parsed_text = re.sub(mention_regex, '', parsed_text)
    return parsed_text

#def tokenize(tweet):
    """Removes punctuation & excess whitespace, sets to lowercase,
    and stems tweets. Returns a list of stemmed tokens."""
    #tweet = " ".join(re.split("[^a-zA-Z]*", tweet.lower())).strip()
    #tokens = [stemmer.stem(t) for t in tweet.split()]
    #return tokens

def tokenize(tweet):
    tk=tt()
  
    #p.set_options(p.OPT.URL, p.OPT.EMOJI)
    tweet=tweet.lower()
    tweet=p.clean(tweet)
    tokens=tk.tokenize(tweet)
    tokens=[w for w in tokens if w[0] not in punctuation]
    tokens = [stemmer.stem(t) for t in tokens]
    tokens=[t for t in tokens if t not in stopwords]
    

    return(tokens)


def basic_tokenize(tweet):
    """Same as tokenize but without the stemming"""
    tweet = " ".join(re.split("[^a-zA-Z.,!?]*", tweet.lower())).strip()
    return tweet.split()

vectorizer = TfidfVectorizer(
    tokenizer=tokenize,
    preprocessor=preprocess,
    ngram_range=(1, 3),
    stop_words=stopwords,
    use_idf=True,
    smooth_idf=False,
    norm=None,
    decode_error='replace',
    max_features=10000,
    min_df=5,
    max_df=0.501
    )

print(stopwords)
# In[9]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[10]:


#Construct tfidf matrix and get relevant scores
tfidf = vectorizer.fit_transform(tweets).toarray()
vocab = {v:i for i, v in enumerate(vectorizer.get_feature_names())}
idf_vals = vectorizer.idf_
idf_dict = {i:idf_vals[i] for i in vocab.values()} #keys are indices; values are IDF scores


# In[11]:


#Get POS tags for tweets and save as a string
tweet_tags = []
for t in tweets:
    tokens = basic_tokenize(preprocess(t))
    tags = nltk.pos_tag(tokens)
    tag_list = [x[1] for x in tags]
    tag_str = " ".join(tag_list)
    tweet_tags.append(tag_str)


# In[12]:


#We can use the TFIDF vectorizer to get a token matrix for the POS tags
pos_vectorizer = TfidfVectorizer(
    tokenizer=None,
    lowercase=False,
    preprocessor=None,
    ngram_range=(1, 3),
    stop_words=None,
    use_idf=False,
    smooth_idf=False,
    norm=None,
    decode_error='replace',
    max_features=5000,
    min_df=5,
    max_df=0.501,
    )


# In[13]:


#Construct POS TF matrix and get vocab dict
pos = pos_vectorizer.fit_transform(pd.Series(tweet_tags)).toarray()
pos_vocab = {v:i for i, v in enumerate(pos_vectorizer.get_feature_names())}


# In[14]:


#Now get other features
sentiment_analyzer = VS()

def count_twitter_objs(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE
    4) hashtags with HASHTAGHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned.
    
    Returns counts of urls, mentions, and hashtags.
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    hashtag_regex = '#[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, 'URLHERE', parsed_text)
    parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)
    parsed_text = re.sub(hashtag_regex, 'HASHTAGHERE', parsed_text)
    return(parsed_text.count('URLHERE'),parsed_text.count('MENTIONHERE'),parsed_text.count('HASHTAGHERE'))

def other_features(tweet):
    """This function takes a string and returns a list of features.
    These include Sentiment scores, Text and Readability scores,
    as well as Twitter specific features"""
    sentiment = sentiment_analyzer.polarity_scores(tweet)
    
    words = preprocess(tweet) #Get text only
    
    syllables = textstat.syllable_count(words)
    num_chars = sum(len(w) for w in words)
    num_chars_total = len(tweet)
    num_terms = len(tweet.split())
    num_words = len(words.split())
    avg_syl = round(float((syllables+0.001))/float(num_words+0.001),4)
    num_unique_terms = len(set(words.split()))
    
    ###Modified FK grade, where avg words per sentence is just num words/1
    FKRA = round(float(0.39 * float(num_words)/1.0) + float(11.8 * avg_syl) - 15.59,1)
    ##Modified FRE score, where sentence fixed to 1
    FRE = round(206.835 - 1.015*(float(num_words)/1.0) - (84.6*float(avg_syl)),2)
    
    twitter_objs = count_twitter_objs(tweet)
    retweet = 0
    if "rt" in words:
        retweet = 1

    lyric=False
    #index=0
    for t in tweets:
        new_t= preprocess(t)
        #print(new_t)
        for l in lyrics:
            l=preprocess(l)
            if new_t == l:
            #print(new_t)
            #print(tweets.iloc[index])
            #df.loc[index, "class"]=2
                lyric=True
                #df.set_value(index,'class',2)
                #df.to_csv("labeled_data.csv", index=False)
                #print("done", df["class"].iloc[index])
        #index=index+1


    features = [FKRA, FRE,syllables, avg_syl, num_chars, num_chars_total, num_terms, num_words,
                num_unique_terms, sentiment['neg'], sentiment['pos'], sentiment['neu'], sentiment['compound'],
                twitter_objs[2], twitter_objs[1],
                twitter_objs[0], lyric]
    #features = pandas.DataFrame(features)
    return features

def get_feature_array(tweets):
    feats=[]
    for t in tweets:
        feats.append(other_features(t))
    return np.array(feats)


# In[15]:


other_features_names = ["FKRA", "FRE","num_syllables", "avg_syl_per_word", "num_chars", "num_chars_total", "num_terms", "num_words", "num_unique_words", "vader neg","vader pos","vader neu", "vader compound", "num_hashtags", "num_mentions", "num_urls","lyric"]


# In[16]:


feats = get_feature_array(tweets)


# In[17]:


#Now join them all up
M = np.concatenate([tfidf,pos,feats],axis=1)


# In[18]:


M.shape


# In[19]:


#Finally get a list of variable names
variables = ['']*len(vocab)
for k,v in vocab.items():
    variables[v] = k

pos_variables = ['']*len(pos_vocab)
for k,v in pos_vocab.items():
    pos_variables[v] = k

feature_names = variables+pos_variables+other_features_names


# # Running the model
# 
# The best model was selected using a GridSearch with 5-fold CV.

# In[20]:



def is_lyric(tweets):
    index=0
    ar=[]
    for t in tweets:
        ar=other_features(t)
        if ar[-1]==True:
            df.set_value(index,'class',2)
            df.to_csv("labeled_data.csv", index=False)
            #print("done", df["class"].iloc[index])
        index=index+1

is_lyric(tweets)

X = pd.DataFrame(M)
y = df['class'].astype(int)


# In[21]:


select = SelectFromModel(LogisticRegression(class_weight='balanced',penalty="l1",C=0.01))
X_ = select.fit_transform(X,y)

model = LinearSVC(class_weight='balanced',C=0.01, penalty='l2', loss='squared_hinge',multi_class='ovr').fit(X_, y)

y_preds = model.predict(X_)

report = classification_report( y, y_preds )

print(report)


pickle.dump(model, open('final_model.pkl',"wb"))

print("I dumped final_model.pkl! Good Luck!")


#################
#Using information from the model to obtain the matrix X_ generically

print("I got here")


final_features = select.get_support(indices=True) #get indices of features
final_feature_list = [str(feature_names[i]) for i in final_features] #Get list of names corresponding to indices

print (final_feature_list)

print("I got hereuuu")

#Getting names for each class of features
ngram_features = final_feature_list[:final_feature_list.index('zimmerman')+1]
pos_features = final_feature_list[final_feature_list.index('zimmerman')+1:final_feature_list.index('FKRA')]
oth_features = final_feature_list[final_feature_list.index('FKRA'):]


print("ngram_features: ", ngram_features)
print("pos_features: ", pos_features)
print("oth_features: ", oth_features)



#ngram features

new_vocab = {v:i for i, v in enumerate(ngram_features)}
new_vocab_to_index = {}
for k in ngram_features:
    new_vocab_to_index[k] = vocab[k]

#Get indices of text features
ngram_indices = final_features[:len(ngram_features)]

new_vectorizer = TfidfVectorizer(
    #vectorizer = sklearn.feature_extraction.text.CountVectorizer(
    tokenizer=tokenize,
    preprocessor=preprocess,
    ngram_range=(1, 3),
    stop_words=stopwords, #We do better when we keep stopwords
    use_idf=False,
    smooth_idf=False,
    norm=None, #Applies l2 norm smoothing
    decode_error='replace',
    min_df=1,
    max_df=1.0,
    vocabulary=new_vocab
    )


pickle.dump(new_vectorizer, open('final_tfidf.pkl',"wb"))

#with open(final_tfidf.pkl, 'wb') as pickle2_file:
    #pickle.dump(new_vectorizer, pickle2_file)

print("I dumped final_tfidf.pkl")

tfidf_ = new_vectorizer.fit_transform(tweets).toarray()
tfidf_[1,:]
tfidf_[1,:].sum()
X_[1,:tfidf_.shape[1]]
X_[1,:tfidf_.shape[1]].sum()
idf_vals_ = idf_vals[ngram_indices]
idf_vals_.shape

#TODO: Pickle idf_vals

#with open(final_idf.pkl, 'wb') as pickle_file:
    #pickle.dump(idf_vals_, pickle_file)


pickle.dump(idf_vals_, open('final_idf.pkl',"wb"))
print("I dumped final_idf.pkl")

(tfidf_[1,:]*idf_vals_) == X_[1,:153] #Got same value as final process array!

tfidf_*idf_vals_ == X_[:,:153]

tfidffinal = tfidf_*idf_vals_

#POS

new_pos = {v:i for i, v in enumerate(pos_features)}

#TODO: Pickle pos vectorizer
#We can use the TFIDF vectorizer to get a token matrix for the POS tags
new_pos_vectorizer = TfidfVectorizer(
    #vectorizer = sklearn.feature_extraction.text.CountVectorizer(
    tokenizer=None,
    lowercase=False,
    preprocessor=None,
    ngram_range=(1, 3),
    stop_words=None, #We do better when we keep stopwords
    use_idf=False,
    smooth_idf=False,
    norm=None, #Applies l2 norm smoothing
    decode_error='replace',
    min_df=1,
    max_df=1.0,
    vocabulary=new_pos
    )

pickle.dump(new_pos_vectorizer, open('final_pos.pkl',"wb"))

#with open(final_pos.pkl, 'wb') as pickle1_file:
    #pickle.dump(new_pos_vectorizer, pickle1_file)

print("I dumped final_pos.pkl")

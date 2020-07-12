#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.cloud import storage
from google.cloud.storage.blob import Blob
import os

import pandas as pd
import numpy as np
from pprint import pprint
import string
import matplotlib.pyplot as plt
import seaborn as sns
from langdetect import detect
import jieba

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim import corpora, models, similarities

import pyLDAvis
import pyLDAvis.gensim


# ## Beijing

# In[2]:


reviews_beijing = pd.read_csv("review_beijing.csv")
reviews_beijing.head()


# In[3]:


df = reviews_beijing.groupby("date")['id'].count().reset_index()
df_covid = df[(df.date>"2020-01-01") & (df.date<"2020-03-01")]


# In[4]:


fig = plt.figure(figsize=(20,8))
plt.xticks(rotation=45)
sns.lineplot(df_covid.date, df_covid.id)


# ### Topics model

# #### 1) comments cleaning

# In[5]:


# load stopwords text file
with open("chinese_stop_words.txt") as f:
    stopwordList = f.readlines()

stopwordList = [x.strip() for x in stopwordList]

punctuations = list(string.punctuation)
# add additional stop words in Chinese
chinese_punct=['｡',"；",'','z','稍', "､", "", "挺", "想", "太", "特别",
               "里", "说", "一点", "两个", "三个", "Airbnb"]
punctuations.extend(chinese_punct)
stopwordList.extend(punctuations)


# In[6]:


# write the function to tokenize comments using jieba
def wordsLst(sentence):
    comment_token = jieba.cut(sentence.strip(), cut_all=False)
    tokens = ''
    for word in comment_token:
        if word not in stopwordList:
            if word != '\t' and word != '':
                tokens += word
                tokens += " "
    return tokens


# In[7]:


tokens_lst = []
for i in reviews_beijing.comments_nonnum:
    if wordsLst(i) != '':
        tokens_lst.append([i, wordsLst(i)])

tokens_lst = [i[1].split(' ')[:-1] for i in tokens_lst]
tokens_lst


# In[8]:


# create word dictionary
id2word = corpora.Dictionary(tokens_lst)
id2word.filter_extremes(no_below=1000, no_above=0.9, keep_n=5000)
corpus = [id2word.doc2bow(sentence) for sentence in tokens_lst]


# In[9]:


# determine the number of topics
coherenceScores = []
models = []
# try the number of topics from 2 to 10
for num_topics in range(2,10,1):
    # build the lad model
    ldaModel = gensim.models.ldamodel.LdaModel(corpus=corpus,
                    id2word=id2word,
                    num_topics=num_topics,
                    random_state=0,
                    update_every=1,
                    chunksize=100,
                    passes=10,
                    alpha='auto',
                    per_word_topics=True)
    models.append(ldaModel)
    # calculate the coherence score
    coherencemodel = CoherenceModel(model=ldaModel, texts=tokens_lst,
                                    dictionary=id2word, coherence='c_v')
    coherenceScores.append(round(coherencemodel.get_coherence(),3))

print(coherenceScores)


# In[10]:


# plot the coherence scores against the number of topics
fig = plt.figure(figsize=(13,8))
x = range(2,10,1)
plt.plot(x, coherenceScores, color="steelblue")
plt.xlabel("Num of Topics")
plt.ylabel("Coherence Score")
plt.legend(("coherenceScores"), loc='best')
plt.show()


# In[11]:


# find the index of the model with the highest coherence scores
best_idx = np.array(coherenceScores).argmax()
optimalModel = models[best_idx]
# print out representative words in each topic
model_topics = optimalModel.show_topics(formatted=False)
pprint(optimalModel.print_topics(num_words=15))


# In[12]:


# visualize the topics
pyLDAvis.enable_notebook()
vis_title = pyLDAvis.gensim.prepare(optimalModel, corpus, id2word)
vis_title


# In[13]:


pyLDAvis.save_html(vis_title, 'beijing_vis.html')


# In[15]:


# Find the most representative sentence in each topic
pd.options.display.max_colwidth = 150
sent_topics_sorted = pd.DataFrame()
domin_sent = airbnb_beijing_dominant_topic.groupby('dominant_topic')

for i, sen in domin_sent:
    sent_topics_sorted = pd.concat([sent_topics_sorted,
                                        sen.sort_values(['contribution_percent'], ascending=False).head(1)],
                                        axis=0)
sent_topics_sorted.reset_index(drop=True, inplace=True)
sent_topics_sorted.columns = ['doc_no','topic_no', "contri_prob", "keywords", "repre_text"]
sent_topics_sorted


# ## San Francisco

# In[18]:


reviews_sf = pd.read_csv("reviews_SF.csv")
reviews_sf.head()


# In[19]:


# The 'shelter-in-place' policy in SF was executed since 17th Mar, 2020
df_sf = reviews_sf.groupby("date")['id'].count().reset_index()
df_covid_sf = df_sf[(df_sf.date>"2020-03-01") & (df_sf.date<"2020-06-07")]


# In[20]:


fig = plt.figure(figsize=(20,8))
plt.xticks(rotation=90)
sns.lineplot(df_covid_sf.date, df_covid_sf.id, color='darkred')


# # Topic modeling for SF

# In[1]:


# Topic modeling for SF
import pandas as pd
import numpy as np
import datetime
from scipy import sparse
# word embedding
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import string
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import gensim
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim
# visualization
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')
import nltk
nltk.download('stopwords')


# In[2]:


review_sf = pd.read_csv("reviews_sf_Jun08.csv.gz")
review_sf.shape  # There are 346k accumulatd reviews in Beijing market


# ### **Let's see whether the hosting experience and topic of the reviews will differ before and after COVID 19**

# #### After COVID 19

# In[3]:


review_sf_covid = review_sf[review_sf["date"]>"2020-02-01"]
review_sf_covid.shape


# In[8]:


review = review_sf_covid["comments"].dropna()

# convert text to lower case
review = review.str.lower()

# words Tokenization
review = review.apply(word_tokenize)

# remove punctuation
punctuations = list(string.punctuation)
punctuations.append('"')
punctuations.append('"')
punctuations.append('’',)

review = review.apply(lambda x:
           [i.strip("".join(punctuations)) for i in x if i not in punctuations])

# remove stop words
stop_words=set(stopwords.words("english"))
newStopWords= ["san","francisco","sf"]
stop_words.update(newStopWords)
review = review.apply(lambda x: [item for item in x if item not in stop_words])
review


# **Topic Modeling**

# In[9]:


# Create Dictionary
rv_dictionary = gensim.corpora.Dictionary(review)

# Filter out tokens that appear in less than 1000 documents (absolute number) or more than 0.9 documents (percents of total corpus size).
#after the above two steps, keep only the first 20000 most frequent tokens.
rv_dictionary.filter_extremes(no_below=1000, no_above=0.9, keep_n=20000)

# Term Document Frequency
corpus = [rv_dictionary.doc2bow(text) for text in review]


# In[14]:


review_sf_lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=rv_dictionary,
                                           num_topics=3,
                                           random_state=0,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

for idx, topic in review_sf_lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))


# In[15]:


# Compute Perplexity: a measure of how good the model is, the lower the better
print('\nPerplexity: ', review_sf_lda_model.log_perplexity(corpus))

# Compute Coherence Score
coherence_model = CoherenceModel(model=review_sf_lda_model, texts=review,
                                           dictionary=rv_dictionary, coherence='c_v')
coherence_ldamodel = coherence_model.get_coherence()
print('\nCoherence Score: ', coherence_ldamodel)


# #### After COVID 19

# In[16]:


review_sf_normal = review_sf[(review_sf["date"]<="2020-02-01")& (review_sf["date"]>"2019-10-01")]
review_sf_normal


# In[17]:


review_nm = review_sf_normal["comments"].dropna()

# convert text to lower case
review_nm = review_nm.str.lower()

# words Tokenization
review_nm = review_nm.apply(word_tokenize)

# remove punctuation
punctuations = list(string.punctuation)
punctuations.append('"')
punctuations.append('"')
punctuations.append('‘')
punctuations.append('‘')
punctuations.append('’')
review_nm = review_nm.apply(lambda x:
           [i.strip("".join(punctuations)) for i in x if i not in punctuations])

# remove stop words
stop_words=set(stopwords.words("english"))

newStopWords= ["san","francisco","sf"]
stop_words.update(newStopWords)
review_nm = review_nm.apply(lambda x: [item for item in x if item not in stop_words])


# In[18]:


# Create Dictionary
nm_dictionary = gensim.corpora.Dictionary(review_nm)

# Filter out tokens that appear in less than 1000 documents (absolute number) or more than 0.9 documents (percents of total corpus size).
#after the above two steps, keep only the first 20000 most frequent tokens.
nm_dictionary.filter_extremes(no_below=1000, no_above=0.9, keep_n=20000)

# Term Document Frequency
corpus_nm = [nm_dictionary.doc2bow(text) for text in review_nm]


# In[19]:


review_sf_lda_model_nm = gensim.models.ldamodel.LdaModel(corpus=corpus_nm,
                                           id2word=nm_dictionary,
                                           num_topics=5,
                                           random_state=0,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True,)

for idx, topic in review_sf_lda_model_nm.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))


# In[20]:


# Compute Perplexity: a measure of how good the model is, the lower the better
print('\nPerplexity: ', review_sf_lda_model_nm.log_perplexity(corpus))

# Compute Coherence Score
coherence_model_nm = CoherenceModel(model=review_sf_lda_model_nm, texts=review_nm,
                                           dictionary=nm_dictionary, coherence='c_v')
coherence_ldamodel = coherence_model_nm.get_coherence()
print('\nCoherence Score: ', coherence_ldamodel)

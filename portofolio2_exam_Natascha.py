#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 15:51:43 2019

@author: martinejorgensen
"""

# Imports
## Task 1
import json
import os
import pandas
import requests
from os import makedirs
from os.path import join, exists
from datetime import date, timedelta

## Task 2
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.probability import FreqDist 

## Task 4
from sklearn.decomposition import LatentDirichletAllocation 
from wordcloud import WordCloud
import matplotlib.pyplot as plt







''' Task 1 ''' 

# Here we download the data

ARTICLES_DIR = join('theguardian', 'collection')
makedirs(ARTICLES_DIR, exist_ok=True)
MY_API_KEY = 'bfeb3d66-1e24-4e0a-8082-1c6ee9e2a7b7'

API_ENDPOINT = 'http://content.guardianapis.com/search'
# We create a dictionary called my_'params' and set the values 
# Boris Johnson will be our focus. Thus, we will import articles in relation to him. 
# for the API's mandatory parameters
my_params = {
    'q' : 'boris+johnson', 
    'from-date': "", 
    'to-date': "",
    'order-by': "newest",
    'show-fields': 'all',
    'page-size': 200,
    'api-key': MY_API_KEY
}

# We choose to download the data with this start- and end date. 
# We have chosen to download only one week of data to narrow down 
# the text corpus.
start_date = date(2019, 9, 1) 
end_date = date(2019, 11, 1)

dayrange = range((end_date - start_date).days + 1)
for daycount in dayrange:
    dt = start_date + timedelta(days=daycount)
    datestr = dt.strftime('%Y-%m-%d')
    fname = join(ARTICLES_DIR, datestr + '.json')
    if not exists(fname):
        print("Downloading", datestr)
        all_results = []
        my_params['from-date'] = datestr
        my_params['to-date'] = datestr
        current_page = 1
        total_pages = 1
        while current_page <= total_pages:
            print("...page", current_page)
            my_params['page'] = current_page
            resp = requests.get(API_ENDPOINT, my_params)
            data = resp.json()
            all_results.extend(data['response']['results'])
            current_page += 1
            total_pages = data['response']['pages']
        with open(fname, 'w') as f:
            print("Writing to", fname)
            f.write(json.dumps(all_results, indent=2))


# Here we read the data 
directory_name = "theguardian/collection/"

ids = list()
strtexts = ""
texts = list()
allfields = list()
allheadlines = list()
allSections = list()


for filename in os.listdir(directory_name): #For ever file in the guardian collection, listdir every file in this collection - it returns an unsorted list of all the directories in the path..
    if filename.endswith(".json"): # If the file ends with json, open the collection as well as filenames as json files. 
        with open(directory_name + filename) as json_file: 
            data = json.load(json_file) #saves the json-files in a variable, names data.
            for article in data: #for every article in the variable data. 
                id = article['id'] # id is defined as the articles ids 
                headline = article['webTitle'] #the variable headline is defined as the html-title, 'webTitle', in the corpus.
                sections = article['sectionName']# etc. 
                fields = article['fields']
                text = fields['bodyText'] if fields['bodyText'] else ""  
                ids.append(id) #appends the variable id to a list
                strtexts += text #adds text to the variable to create a string version
                texts.append(text) # creates a list version of all the texts
                allSections.append(sections) 
                allfields.append(fields)
                allheadlines.append(headline)

print("Number of ids:", len(ids))
print("Number of texts:", len(texts), "Number of fields", len(allfields))






''' Task 2 ''' 

# We have generated a dataframe for a better overview of the dataset

dataframe = pd.DataFrame({'all_fields': allfields, 
                          'all_headlines': allheadlines, 
                          'all_texts': texts,
                          'section_ids': allSections})
dataframe.shape # (13713 rows and 5 columns)


# We will derive a document-term matrix for our collection. 
# We remove stopwords and count the amount of words, 
# and lastly convert the data to a matrix. 

countvect = CountVectorizer(min_df = 1, 
                            stop_words = stopwords.words('english'), 
                            token_pattern = r'[a-zA-Z\-][a-zA-Z\-]{2,}') 
vecfit = countvect.fit_transform(texts) # transforming data to matrix, vectors.

# We print a list over terms from the variable, texts, sorted according to their index, which is produced by fit_transform
matrixwords = countvect.get_feature_names()
print(matrixwords) 
matrixx = vecfit.toarray() 
print(matrixx) 
# Here we print the shape of our matrix 
print(vecfit.shape) #1669, 43713

# Show words and index numbers
# Access the entire vocabulary to see what exactly was tokenized by calling
top_ordindex=countvect.vocabulary_
print(top_ordindex) 

# Amount of words in new vocabulary - only unique words
print(len(countvect.vocabulary_)) 

# Calculating the matrix sparsity
sparsity = 1.0- np.count_nonzero(matrixx) / matrixx.size
print(sparsity)

# Result: 0.99. 
# Thus, our matrix is sparse as its sparsity is greater than 0.5.


# TF-IDF weighting. 
# Calculating the weight of the words by using scikit's tfidtransform 
# on our previous document-term count matrix.
tfidfmodel = TfidfTransformer()
datafittransformer = tfidfmodel.fit_transform(vecfit)
print(datafittransformer.shape)
print(datafittransformer.toarray()) 


#How many documents = 1669
len(texts)


# Pre-processing
# Word count before and after pre-processing
# Tokenize loop

# We ensure that the text (as a string) is divided into words by using word_tokenize
tokenizedtext = word_tokenize(strtexts)
no_stopwords = []

# Then, we save the stopwords in a variable 
stop_words = set(stopwords.words('english')) 


# And create a loop, that goes thorugh all the words in the tokenized text and checks if it contains stopwords. If not, it will be saved to a list. 
for bestemtord in tokenizedtext: 
    if bestemtord.lower() not in stop_words: 
        if bestemtord.isalpha():
            no_stopwords.append(bestemtord)

print("Previous word count: ", len(tokenizedtext)) #2330313
print("Word count (removed stopwords and tokens): ", len(no_stopwords)) #1055036

# Average length of documents = 632.1 words - after tokenization
len(no_stopwords)/len(texts)  


# Top 20 most common words in our collection. 'Johnson' and 'Brexit' are very popular words. 
ferquencywords = FreqDist(no_stopwords)
top_topwords = ferquencywords.most_common(20)
print(len(top_topwords)) # gets total amount of topics
print(top_topwords) 


# Distribution of articles in different topics
ferquencynewsgroups = FreqDist(dataframe['section_ids'])
top_newsgroups = ferquencynewsgroups.most_common(30)
print(len(top_newsgroups)) # gets total amount of topics
print(top_newsgroups) 






''' Task 3 '''

#First, we create a query focusing on politics to explore the current agenda in relation to Boris Johnson.
terms = ['Politics']
terms

query = " ".join(terms)
query

# Then we create a matrix of our query, countvect
query_vect_counts = countvect.transform([query])
query_vect = tfidfmodel.transform(query_vect_counts)
query_vect






''' Task 4 '''
# We generated a topicmodel with one component since we are only focusing on politics in relation to Boris Johnson (whom we specified within our search parameters in the beginning) 
# and make it replicable to ensure that the results will be the same each time we run the code. 
topicModel_lda = LatentDirichletAllocation(n_components=1, random_state=0) 
# We use our previous document-term count matrix, vecfit. 
data_lda = topicModel_lda.fit_transform(vecfit) 
np.shape(data_lda)
print(data_lda)

#We sort the term weights according to the 10 most popular words in a loop. 
#then we get the topwords from countvect.getfeaturenames to show the words. 
for i, term_weights in enumerate(topicModel_lda.components_):
    top_idxs = (-term_weights).argsort()[:10]
    top_words = ["%s (%.3f)" % (countvect.get_feature_names()[idx], term_weights[idx]) for idx in top_idxs]
    print("Topic %d: %s" % (i, ", ".join(top_words)))

#We included a wordcloud to present our results.
#This invovled creating a new variable that contained a dictionary of topwords and termweights. 
#Then our wordcloud can show the most popular words according to their weight. 
for i, term_weights in enumerate(topicModel_lda.components_):
    top_idxs = (-term_weights).argsort()[:10]
    top_words = [countvect.get_feature_names()[idx] for idx in top_idxs]
    word_freqs = dict(zip(top_words, term_weights[top_idxs]))
    wc = WordCloud(background_color="white",width=1000,height=1000, max_words=10).generate_from_frequencies(word_freqs)
    plt.subplot(1, 1, i+1)
    plt.imshow(wc)


# Thus, it becomes evident that topics regarding brexit, government and deal have on the agenda during these last months. 
# These comply with reality as they are in fact current and well-debated topics in the UK. 



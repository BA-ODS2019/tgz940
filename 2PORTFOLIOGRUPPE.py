#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 15:51:43 2019

@author: martinejorgensen
"""

#Opgave 1 


import json
import os
import pandas
import requests
from os import makedirs
from os.path import join, exists
from datetime import date, timedelta



#HER DOWNLOADE DATAEN

ARTICLES_DIR = join('theguardian', 'collection')
makedirs(ARTICLES_DIR, exist_ok=True)
MY_API_KEY = 'bfeb3d66-1e24-4e0a-8082-1c6ee9e2a7b7'

API_ENDPOINT = 'http://content.guardianapis.com/search'
my_params = {# We create a dictionary called my_'params' and set the values for the API's mandatory parameters
    'from-date': "", 
    'to-date': "",
    'order-by': "newest",
    'show-fields': 'all',
    'page-size': 200,
    'api-key': MY_API_KEY
}

start_date = date(2019, 9, 1) # we choose to download the data with this start- and end date. We have chosen to doenload only one week of data to narrow down the text corpus. 
end_date = date(2019, 9, 7)

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


#Here we read the data 
directory_name = "theguardian/collection/"


ids = list()
strtexts = ""
texts = list()
allfields = list()
allheadlines = list()
alldates= list()
allSections =list()

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
                artcdate = article['webPublicationDate']
                ids.append(id) #appends the variable id to a list
                strtexts+=text #adds text to the variable to create a string version
                texts.append(text) # creates a list version of all the texts
                allSections.append(sections) 
                allfields.append(fields)
                allheadlines.append(headline)
                alldates.append(artcdate)

print("Number of ids:", len(ids))
print("Number of texts:", len(texts), "Number of fields", len(allfields))














#Opgave 2
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer



#we have generated a dataframe for a better overview of the dataset
dataframe = pd.DataFrame({'all_fields':allfields, 'all_headlines': allheadlines, 'all_texts': texts,'date': alldates, 'section_ids': allSections})
dataframe.shape #(1467 rows and 5 columns)


#Derive a document-term matrix for your collection. 
#We remove stopwords and count the amount of words, and lastly convert the data to a matrix. 
countvect = CountVectorizer(stop_words= stopwords.words('english'), token_pattern=r'[a-zA-Z\-][a-zA-Z\-]{2,}') #tæller antal ord-forekomster i datasættet. 
vecfit = countvect.fit_transform(texts) # transformerer data til matrix, vektorer. 

#Printing matrix 
matrixwords = countvect.get_feature_names()
print(matrixwords) #returnerer liste over termerne, sorteret efter deres indeks - som transform har produceret. 
matrixx = vecfit.toarray() 
print(matrixx) #matrix, viser ordenes forekomster i teksterne.

print(vecfit.shape)# rækker: tekster, 1467  og  Kolonner: ord+forekomst 49 907
top_ordindex=countvect.vocabulary_
print(top_ordindex) # #får ord og indexnumre. access the entire vocabulary to see what exactly was tokenized by calling
print(len(countvect.vocabulary_))  


#Calculating the matrix sparsity
from numpy import count_nonzero
sparsity = 1.0- count_nonzero(matrixx) / matrixx.size
print(sparsity) #Result: 0.9967162066385401. Thus, our matrix is sparse -> a matrix will be sparse when its sparsity is greater than 0.5.



#TF-IDF weighting. Calculating the weight of the words by using scikit's tfidtransform on our previous document-term count matrix.
from sklearn.feature_extraction.text import TfidfTransformer
tfidfmodel = TfidfTransformer()
datafittransformer =tfidfmodel.fit_transform(vecfit)
print(datafittransformer.shape)
print(datafittransformer.toarray()) 

#How many documents = 1467
len(texts)

#Average length of documents = 34.01976823449216 words after tokenization
len(countvect.vocabulary_)/len(texts)  

#Pre-processing
#Word count before and after pre-processing
tokenizedtext = word_tokenize(strtexts) #ensures that the text is devided into words 
print("Previous total word count: ", len(tokenizedtext))
print("Total word count now: ", len(countvect.vocabulary_))


#Remove stopwords from tokenzied texts to create a random sample
import random
random.sample(list(countvect.vocabulary_), 10)

#Frequency distribution of the list of unique tokens. 
from nltk.probability import FreqDist 
frekvens = FreqDist(countvect.vocabulary_)  #we use the countvect vocabulary, since it contains the words that we have tokenized and preprocessed. 
frekvens.most_common(20) 


#Distribution of articles in different topics
from nltk.probability import FreqDist 
frekvens = FreqDist(dataframe['section_ids'])
top_words = frekvens.most_common(100) 


#Calculating amount of articles/documents for a specific date, eg. there is 176 articles from the date: "2019-09-07"
ny = list()
listedf = list(dataframe.date)
for i in range(len(dataframe.date)):
    if listedf[i][0:10] == "2019-09-07": 
        ny.append(listedf[i])
print(ny)
len(ny)


#Calculates the average word length = 8.06
vocabulary=list(countvect.vocabulary_)
total =0
for token in range(len(vocabulary)): #hvor hvert ord i listen tokens, der indeholder pre processed ord fra datasættet.
    total+= len(list(vocabulary)[token]) #tilføjer deres længde til en counter
average= total/len(vocabulary) #total delt med antal ord
print(average)














#Opgave 3 
counts = vecfit.sum(axis=0).A1  #laver variabel, tæller, summen af X listen, aksen er  0, lægger alle elementerne sammen, A1 giver array. antal indexnumre hvori besteme termer har optrådt . 
top_idxs = (-counts).argsort()[:500] #Returns the indices that would sort an array. til og med 10 
top_idxs 

inverted_vocabulary = dict([(idx, word) for word, idx in countvect.vocabulary_.items()]) #index og ord, for hvert ord  og index i vectorizer_vocabulary_items tilføj ord til index i næste variabel. 
top_words = [inverted_vocabulary[idx] for idx in top_idxs]
print("Top words: %s" % top_words)

#Creates a randomized sample consisting of 500 documents
some_row_idxs = random.sample(range(0,len(dataframe['section_ids'])), 100) #laver random sample for 10 dokumenter. Range 0 til totale antal dokumenter. Sample tager 10 vilkårlige inden for antal dokumenter. Giver nogle række-idexer 
some_row_idxs.sort()
sample = some_row_idxs

#Creates submatrix
print("Selection: (%s x %s)" % (some_row_idxs, top_idxs)) # #slicer den ene dimension og dernæst den anden. X og dernæst top10 indexes.
sub_matrix = vecfit[some_row_idxs, :][:, top_idxs].todense() #  X, starter fra de første 10 vilkårlige dokumente-rækker, slutter ved top10 indexes. Tager en slice af matrixen. 
sub_matrix 

#Dataframen
import pandas as pd
df = pd.DataFrame(columns=top_words, index=some_row_idxs, data=sub_matrix) # kolonner: top ordene, index er de vilkårlige dokument rækker, data er den slicede matrix. 
df.insert(0, 'Sektion', dataframe['section_ids'][sample]) #target variablen, værdierne, hvilken nyhedsgruppe, det tilhører. Hvilke ofte forekommer ofte inden for de forskellige nyhedsgrypper. Fx. der er to 8, der er to dokumenter inden for den samme nyhedsgruppe. 
print(df)


alle = list()
#Indexnmuber of the articles within the section of politics in our random sample
dfpoliticsIndex = list()
for i in range(len(sample)): 
    if df['Sektion'][sample[i]] == 'Politics':
        dfpoliticsIndex.append(sample[i])
        print(sample[i],'Section: ', df['Sektion'][sample[i]], ' headline: ', dataframe['all_headlines'][sample[i]], '\n')



#Here we generate a groupby table of the df-dataframe where we focus on Johnson and the section, politics to narrow down our query.
#We can observe that the news section, where johnson is most discussed, is politics, within our random sample of 100 documents. He has been mentioned various times as indicated in the word counts on the right side of the table. 
ny_df = df.groupby('Sektion')['johnson'].apply(list)
list(ny_df)
print(ny_df)



#We have created a subdataframe where we use the index numbers of the articles within politics and the groupby table, ny_df
#in order to find the excact index position of politics within the ny_df, we have to manually count its position when printing ny_df
#limit: in order for the new dataframe to work, it is necessary to manually count indexposition of politics in ny_df.
dflistzip = list(zip(ny_df[13],dfpoliticsIndex))
dataframesub = pd.DataFrame(dflistzip, columns = ['johnson', 'artikel']) 















#Opgave 4

from sklearn.decomposition import LatentDirichletAllocation 
topicModel_lda = LatentDirichletAllocation(n_components=4, random_state=0) # 4 components, random=0 (reproducerbar)
data_lda = topicModel_lda.fit_transform(vecfit)
import numpy as np
np.shape(data_lda)
print(data_lda)


for i, term_weights in enumerate(topicModel_lda.components_):
    top_idxs = (-term_weights).argsort()[:10]
    top_words = ["%s (%.3f)" % (countvect.get_feature_names()[idx], term_weights[idx]) for idx in top_idxs]
    print("Topic %d: %s" % (i, ", ".join(top_words)))


from wordcloud import WordCloud
import matplotlib.pyplot as plt

for i, term_weights in enumerate(topicModel_lda.components_):
    top_idxs = (-term_weights).argsort()[:10]
    top_words = [countvect.get_feature_names()[idx] for idx in top_idxs]
    word_freqs = dict(zip(top_words, term_weights[top_idxs]))
    wc = WordCloud(background_color="white",width=300,height=300, max_words=10).generate_from_frequencies(word_freqs)
    plt.subplot(2, 2, i+1)
    plt.imshow(wc)









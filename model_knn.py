#!/usr/bin/env python
# coding: utf-8

# In[33]:


import numpy as np
import pandas as pd
import re
import nltk
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pickle

# from wordcloud import WordCloud, STOPWORDS


# In[2]:


# https://www.kaggle.com/c/nlp-getting-started/overview


# In[3]:


# Reading training and testing datasets
ds_train = pd.read_csv("Dataset/train.csv")
ds_test = pd.read_csv("Dataset/test.csv")

# In[4]:


# printing training dataset
ds_train.head()
print(len(ds_train))

# In[5]:


# Calculating NaN values.
train_nans = ds_train['keyword'].isnull().sum()
print(train_nans)
train_nans = ds_train['location'].isnull().sum()
print(train_nans)
test_nans = ds_test['keyword'].isnull().sum()
print(test_nans)
test_nans = ds_test['location'].isnull().sum()
print(test_nans)

# In[6]:


# Replacing nans.
ds_train['keyword'].fillna('none', inplace=True)
ds_train['location'].fillna('Unknown', inplace=True)
ds_test['keyword'].fillna('none', inplace=True)
ds_test['location'].fillna('Unknown', inplace=True)

# In[7]:


# Checking NaN values again.
ds_train.head()
train_nans = ds_train['keyword'].isnull().sum()
print(train_nans)
train_nans = ds_train['location'].isnull().sum()
print(train_nans)

# In[8]:


# Creating Corpus after preprocessing the training data
corpus = []
pstem = PorterStemmer()
for i in range(ds_train['text'].shape[0]):
    text = re.sub("[^a-zA-Z]", ' ', ds_train['text'][i])
    text = text.lower()
    text = text.split()
    text = [pstem.stem(word) for word in text if not word in set(stopwords.words('english'))]
    text = ' '.join(text)
    corpus.append(text)

# In[9]:


corpus_test = []
pstem = PorterStemmer()
for i in range(ds_test['text'].shape[0]):
    text = re.sub("[^a-zA-Z]", ' ', ds_train['text'][i])
    text = text.lower()
    text = text.split()
    text = [pstem.stem(word) for word in text if not word in set(stopwords.words('english'))]
    text = ' '.join(text)
    corpus_test.append(text)

# In[10]:


# Create dictionary based on corpus
uniqueWords = {}
for text in corpus:
    for word in text.split():

        if (word in uniqueWords.keys()):
            uniqueWords[word] = uniqueWords[word] + 1
        else:
            uniqueWords[word] = 1

# In[11]:


traindf = pd.DataFrame()

# In[12]:


for i in list(uniqueWords.keys()):
    tem = []
    for j in corpus:
        if i in j:
            tem.append(1)
        else:
            tem.append(0)
    traindf[str(i)] = tem

# In[13]:


testdf = pd.DataFrame()

# In[14]:


for i in list(uniqueWords.keys()):
    tem = []
    for j in corpus_test:
        if i in j:
            tem.append(1)
        else:
            tem.append(0)
    testdf[str(i)] = tem

# In[15]:


from sklearn.neighbors import KNeighborsClassifier

classifier_knn = KNeighborsClassifier(n_neighbors=6, weights='distance', algorithm='brute')
classifier_knn.fit(traindf, ds_train['target'])
y_pred_knn = classifier_knn.predict(testdf)

# In[34]:


pickle.dump(classifier_knn, open('model_knn.pkl', 'wb'))

# In[16]:


y_pred_knn

# In[17]:


resultdf = pd.DataFrame({'ID': ds_test['id'], 'target': y_pred_knn})
resultdf.to_csv(r'Dataset/MT21149_test_result_IR_knn.csv', index=False)

# In[18]:


# uniqueWords_test = {}
# for text in corpus_test:
#     for word in text.split():
#         if(word in uniqueWords.keys() and word not in uniqueWords_test.keys()):
#             uniqueWords_test[word] = 1
#         elif(word in uniqueWords.keys()):
#             uniqueWords_test[word] = uniqueWords_test[word] + 1
#         else:
#             uniqueWords_test[word] = 1


# In[19]:


# print(len(uniqueWords_test))


# In[20]:


# for i in uniqueWords.keys():
#     if(i not in uniqueWords_test.keys()):
#         uniqueWords_test[i] = 0


# In[21]:


# #Converting dictionary to dataFrame
# uniqueWords = pd.DataFrame.from_dict(uniqueWords,orient='index',columns=['WordFrequency'])
# uniqueWords.sort_values(by=['WordFrequency'], inplace=True, ascending=False)
# #print(uniqueWords)
# print("Number of records in Unique Words Data frame are ")
# print((len(uniqueWords)))
# uniqueWords.head(10)


# In[22]:


# #Converting dictionary to dataFrame
# uniqueWords_test = pd.DataFrame.from_dict(uniqueWords_test,orient='index',columns=['WordFrequency'])
# uniqueWords_test.sort_values(by=['WordFrequency'], inplace=True, ascending=False)
# #print(uniqueWords)
# print("Number of records in Unique Words Data frame are ")
# print((len(uniqueWords_test)))
# uniqueWords_test.head(10)


# In[23]:


# unique_len= len(uniqueWords)


# In[24]:


# #Creating the Bag of Words model by vectorizing the input data
# from sklearn.feature_extraction.text import CountVectorizer
# cv = CountVectorizer(max_features = unique_len)
# X = cv.fit_transform(corpus).todense()
# y = ds_train['target'].values


# In[25]:


# unique_len= len(uniqueWords_test)


# In[26]:


# # Creating the Bag of Words model by vectorizing the input data
# from sklearn.feature_extraction.text import CountVectorizer
# cv_test = CountVectorizer(max_features = unique_len)
# X_test = cv.fit_transform(corpus_test).todense()


# In[27]:


# corpus_test


# In[28]:


# #Split the training data set to train and test data
# X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2)
# print('Train Data splitted successfully')
# print(X_test)


# In[29]:


# X_train=X
# y_train=y


# In[30]:


# from sklearn.neighbors import KNeighborsClassifier


# In[31]:


# # Applying Model KNN
# classifier_knn = KNeighborsClassifier(n_neighbors = 6,weights = 'distance',algorithm = 'brute')
# classifier_knn.fit(X_train[:3263], y_train[:3263])
# y_pred_knn = classifier_knn.predict(X_test)


# In[ ]:


# #Calculating Evaluation Measures
# print("K-Nearest Neighbour Model Accuracy Score for Train Data set is")
# print((classifier_knn.score(X_train, y_train)))
# print("K-Nearest Neighbour Model Accuracy Score for Test Data set is ")
# print(classifier_knn.score(X_test, y_test) )   
# print("K-Nearest Neighbour Model F1 Score is ")
# print((f1_score(y_test, y_pred_knn)))


# In[ ]:


# X_test


# In[ ]:


# var = input("Please enter the news text you want to verify: ")
# print("You entered: " + str(var))


# In[ ]:


# # Creating Corpus after preprocessing the training data
# test  = []
# pstem = PorterStemmer()

# text = re.sub("[^a-zA-Z]", ' ', var)
# text = text.lower()
# text = text.split()
# text = [pstem.stem(word) for word in text if not word in set(stopwords.words('english'))]
# text = ' '.join(text)
# test.append(text)


# In[ ]:


# test


# In[ ]:


# #Create dictionary based on corpus 
# uniqueWords = {}
# for text in corpus:
#     for word in text.split():

#         if(word in uniqueWords.keys()):
#             uniqueWords[word] = uniqueWords[word] + 1
#         else:
#             uniqueWords[word] = 1


# In[ ]:


# unique_len

# In[ ]:


# #Converting dictionary to dataFrame
# uniqueWords = pd.DataFrame.from_dict(uniqueWords,orient='index',columns=['WordFrequency'])
# uniqueWords.sort_values(by=['WordFrequency'], inplace=True, ascending=False)
# #print(uniqueWords)
# print("Number of records in Unique Words Data frame are ")
# print((len(uniqueWords)))
# uniqueWords


# In[ ]:


# # Creating the Bag of Words model by vectorizing the input data
# from sklearn.feature_extraction.text import CountVectorizer
# cv = CountVectorizer(max_features = unique_len)
# X = cv.fit_transform(test).todense()


# In[ ]:


# X=[]


# In[ ]:


# for i in test:
#     for j in uniqueWords:
#         if i in j:
#             X.append(1)
#         else:
#             X.append(0)


# In[ ]:


# X = cv.fit_transform(X).todense()


# In[ ]:


# y_pred_knn = classifier_knn.predict(X)

# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# Creating the Bag of Words model by vectorizing the input data
# from sklearn.feature_extraction.text import CountVectorizer
#
# cv = CountVectorizer(max_features=len(uniqueWords))
# X = cv.fit_transform(corpus).todense()
# y = ds_train['target'].values

# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:





from pyvi import ViTokenizer, ViPosTagger
import numpy as np
import pandas as pd
import gensim
import sklearn
import tensorflow as tf
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.layers import *
from keras.layers import *
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.metrics import sparse_categorical_accuracy
from tensorflow.keras.optimizers import Adam
datanewscontent=pd.read_csv(r'/content/drive/MyDrive/TaskA/TaskA-TrainingSet.csv')
datanews=[]
for d in datanewscontent['dialogue']:
    e=ViTokenizer.tokenize(str(d))
    datanews.append(e)
labelnews=datanewscontent['section_header']
datanewscontentv=pd.read_csv(r'/content/drive/MyDrive/TaskA/TaskA-ValidationSet.csv')
datanewsv=[]
for dv in datanewscontentv['dialogue']:
    ev=ViTokenizer.tokenize(str(dv))
    datanewsv.append(ev)
labelnewsv=datanewscontentv['section_header']
datanewscontentt=pd.read_csv(r'/content/drive/MyDrive/TaskA/TaskA-TestSet.csv')
datanewst=[]
for dt in datanewscontentt['dialogue']:
    et=ViTokenizer.tokenize(str(dt))
    datanewst.append(et)
def truncatedvectors(data,n_components=100):
  svd_ngram = TruncatedSVD(n_components=n_components, random_state=42)
  svd_ngram.fit(data)
  return svd_ngram.transform(data)

def tfidf(data):
  tfidf_vect_ngram = TfidfVectorizer(analyzer='word', max_features=30000, ngram_range=(1, 2))
  tfidf_vect_ngram.fit(data)
  X_data_tfidf_ngram =  tfidf_vect_ngram.transform(data)
  return truncatedvectors(X_data_tfidf_ngram)
X_data_tfidf_news=tfidf(datanews)
X_data_tfidf_newsv=tfidf(datanewsv)
X_data_tfidf_newst=tfidf(datanewst)
import os
dirpath = "/content/drive/MyDrive/TaskA/"
clf=svm.SVC(kernel='linear',C=1000)
clf.fit(X_data_tfidf_news,labelnews)
val_predictions = clf.predict(X_data_tfidf_newst)
print(X_data_tfidf_newsv)
print(val_predictions)
# Importing library
import csv
 
# data to be written row-wise in csv file
#data = [['Geeks'], [4], ['geeks !']]
 
# opening the csv file in 'w+' mode
file = open('/content/drive/MyDrive/TaskA/run1.csv', 'w+', newline ='')
 
# writing the data into the file
with file:   
    write = csv.writer(file)
    i = 0
    for row in val_predictions:
      write.writerow([i,row])
      i += 1
    #write.writerows(val_predictions)



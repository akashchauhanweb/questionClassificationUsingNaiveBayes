# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 21:01:29 2020

@author: akashweb
"""

#importing all the necessary libraries
import string
from sklearn.metrics import f1_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

#creating raw dataset from the text file
rawData = dict()
fh = open('labelledData.txt')
for lines in fh:
    typeSplit = lines.split(',,,')
    if typeSplit[1].strip() in rawData.keys():
        rawData[typeSplit[1].strip()].append(typeSplit[0].strip())
    else:
        rawData[typeSplit[1].strip()] = [typeSplit[0].strip()]

#statistic funcion to check imbalances in categories
def showStatistics(data):
    for typeOfQuestion, sentences in data.items():
        
        noOfSentences = 0
        noOfWords = 0
        noOfUniqueWords = 0
        sampleExtract = ''
        
        words = ' '.join(sentences).split()
        
        noOfSentences = len(sentences)
        noOfWords = len(words)
        noOfUniqueWords = len(set(words))
        sampleExtract = ' '.join(sentences[0].split())
        
        print('Type of question:',typeOfQuestion)
        print('No of sentences:', noOfSentences)
        print('No of words:', noOfWords)
        print('No of unique words:', noOfUniqueWords)
        print('Sample Extract:', sampleExtract,'\n')

#checking imbalancess in raw data        
showStatistics(rawData)

#function for preprocessing or cleaning data
def preprocess(text):
    preprocessedText = text.replace('-',' ')
    translationTable = str.maketrans('\n',' ', string.punctuation+string.digits)
    preprocessedText = preprocessedText.translate(translationTable)
    return preprocessedText

#creating the data structure for preprocessed training data and checking imbalances
preprocessedData = {k: [preprocess(sentence) for sentence in v] for k, v in rawData.items()}
showStatistics(preprocessedData)

#building the training data set from preprocessed training data
sentencesTrain, yTrain = [], []
for k, v in preprocessedData.items():
    for sentence in v:
        sentencesTrain.append(sentence)
        yTrain.append(k)

#vectorizing training data
vectorizer = CountVectorizer()
xTrain = vectorizer.fit_transform(sentencesTrain)

#creating the Multinomial Classifier
naiveClassifier = MultinomialNB()
naiveClassifier.fit(xTrain, yTrain)

#creating the testing data set from text file
valData = dict()
fh = open('testData.txt')
for lines in fh:
    typeSplit = lines.split(',,,')
    if typeSplit[1].strip() in valData:
        valData[typeSplit[1].strip()].append(typeSplit[0].strip())
    else:
        valData[typeSplit[1].strip()] = [typeSplit[0].strip()]

#creating the data structure for preprocessed testing data
preprocessedValData = {k: [preprocess(sentence) for sentence in v] for k, v in valData.items()}

#building the testing data set from preprocessed testing data
sentencesVal, yVal = [], []
for k, v in preprocessedValData.items():
    for sentence in v:
        sentencesVal.append(sentence)
        yVal.append(k)

#vectorizing testing data
xVal = vectorizer.transform(sentencesVal)

#prediciting values for the testing data
predictions = naiveClassifier.predict(xVal)

#plotting confusion matrix to compare predicted data and getting the f1 score
plot_confusion_matrix(naiveClassifier, xVal, yVal, ['unknown','affirmation','what','when','who'])
f1_score(yVal, predictions, average = 'weighted')

#rebuilding classifier to achieve greater efficiency
naiveClassifier = MultinomialNB(alpha=0.5, fit_prior=False)
naiveClassifier.fit(xTrain, yTrain)

#predicting the data again
predictions = naiveClassifier.predict(xVal)

#plotting the confusion matrix again to compare predicted data and getting the f1 score again
plot_confusion_matrix(naiveClassifier, xVal, yVal, ['unknown','affirmation','what','when','who'])
f1_score(yVal, predictions, average = 'weighted')

#getting the user input and letting them check their own
while True:
    sampleString = input('Enter a question - type \'done\' if finished: ')
    if sampleString == 'done':
        break
    sampleString = preprocess(sampleString.lower())
    xVal = vectorizer.transform([sampleString])
    print(naiveClassifier.predict(xVal))
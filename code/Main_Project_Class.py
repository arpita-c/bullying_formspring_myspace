import pandas as pd
import numpy as np
import pdb
import plotly.plotly as py
import plotly.graph_objs as go
import csv
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import TweetTokenizer
from stop_words import get_stop_words
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import enchant
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Perceptron
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import wordnet
import sys

train_documents=[]
test_documents = []
test_documents2 = []
def preprocess(train_file, test_file):

	#
	# training dataset
	#
	header = ['text', 'answer']

	target = []

	d = enchant.Dict("en_US")

	tokenizer = RegexpTokenizer(r'\w+')
	en_stop = get_stop_words('en')
	stop_word = stopwords.words('english')
	p_stemmer = PorterStemmer()

	with open(train_file) as csvfile:
		readCSV = csv.reader(csvfile, delimiter = ",")
		for row in readCSV:
			text = row[0]
			if row[1] == 'yes':
				bullying = 1
			else:
				bullying = 0
			
			tweet = text.strip().lower()
			tokens = tokenizer.tokenize(tweet)
			stopped_tokens = [i for i in tokens if (i not in en_stop) and (i not in stop_word)
							  and (d.check(i))]
			stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
			clean_tweet = ("".join([" " + i for i in stemmed_tokens])).strip()

			train_documents.append(clean_tweet)
			target.append(bullying)
	target = np.array(target)

	#
	# test dataset
	#
	
	
	indexes = []

	with open(test_file) as csvfile:
		readCSV = csv.reader(csvfile, delimiter = "\\")

		for row in readCSV:
			text = row[0]
			if row[1] == 'yes':
				bullying = 1
			else:
				bullying = 0
	
			tweet = text.strip().lower()
			tokens = tokenizer.tokenize(tweet)
			stopped_tokens = [i for i in tokens if (i not in en_stop) and (i not in stop_word)
						  
						  and (d.check(i))]

			stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
			clean_tweet = ("".join([" " + i for i in stemmed_tokens])).strip()
	
			test_documents.append(clean_tweet)
			indexes.append(bullying)

	indexes = np.array(indexes)		

	all_documents = train_documents+ test_documents

	no_features = 1300
	tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
	tf = tf_vectorizer.fit_transform(all_documents)
	tf_feature_names = tf_vectorizer.get_feature_names()

	tf_dense = tf.toarray()
	train_feature = tf_dense[0:len(train_documents),:]
	test_feature = tf_dense[len(train_documents):,:]

	return train_feature,test_feature,target,indexes



def preprocess_bullyingwords(orig_file):

	header = ['text', 'answer','bullyingword']

	target = []
	bullying_documents = []

	d = enchant.Dict("en_US")

	tokenizer = RegexpTokenizer(r'\w+')
	en_stop = get_stop_words('en')
	stop_word = stopwords.words('english')
	p_stemmer = PorterStemmer()

	with open(orig_file) as csvfile:
		readCSV = csv.reader(csvfile, delimiter = ",")
		for row in readCSV:
			text = row[2]
			
			if(text=="na" or text =="None"):
				continue
			tweet = text.strip().lower()
			tokens = tokenizer.tokenize(tweet)
			stopped_tokens = [i for i in tokens if (i not in en_stop) and (i not in stop_word)
							  and (d.check(i))]
			final_words=[]
			for word in stopped_tokens:
				final_words.append(word)
				synonyms=[]
				for syn in wordnet.synsets(word):
					for l in syn.lemmas():
						synonyms.append(l.name())
						final_words.append(l.name())
        	  


			stemmed_tokens = [p_stemmer.stem(i) for i in final_words]
			clean_tweet = ("".join([" " + i for i in stemmed_tokens])).strip()

			bullying_documents.append(clean_tweet)

	no_features = 1300
	tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
	tf = tf_vectorizer.fit_transform(bullying_documents)
	tf_feature_names = tf_vectorizer.get_feature_names()

	tf_dense = tf.toarray()
	train_feature = tf_dense[0:len(train_documents),:]
	test_feature = tf_dense[0:len(test_documents),:]

	return train_feature,test_feature

def preprocess_test_file(test_file):

	#
	# training dataset
	#
	header = ['text']

	d = enchant.Dict("en_US")

	tokenizer = RegexpTokenizer(r'\w+')
	en_stop = get_stop_words('en')
	stop_word = stopwords.words('english')
	p_stemmer = PorterStemmer()

	with open(test_file) as csvfile:
		readCSV = csv.reader(csvfile, delimiter = "\n")

		for row in readCSV:
			if (len(row) == 0):
				pass
			else:
				text = row[0]
	
				tweet = text.strip().lower()
				tokens = tokenizer.tokenize(tweet)
				stopped_tokens = [i for i in tokens if (i not in en_stop) and (i not in stop_word)
							  		and (d.check(i))]

				stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
				clean_tweet = ("".join([" " + i for i in stemmed_tokens])).strip()
		
				test_documents2.append(clean_tweet)

	no_features = 1300
	tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
	tf = tf_vectorizer.fit_transform(test_documents2)
	tf_feature_names = tf_vectorizer.get_feature_names()

	tf_dense = tf.toarray()
	test_feature = tf_dense[0:len(test_documents2):,:]

	return test_feature

def train_logistic_regression(train_x, train_y):
	"""
	Training logistic regression model with train dataset features(train_x) and target(train_y)
	:param train_x:
	:param train_y:
	:return:
	"""

	#logistic_regression_model = LogisticRegression(solver='saga', max_iter=1000, n_jobs=-1)
	logistic_regression_model = LogisticRegression(max_iter=10000)
	logistic_regression_model.fit(train_x, train_y)
	return logistic_regression_model

def train_MultinomialNB(train_x, train_y):
	"""
	Training Multinomial Naive Bayes model with train dataset features(train_x) and target(train_y)
	:param train_x:
	:param train_y:
	:return:
	"""

	MultinomialNB_model = MultinomialNB()
	MultinomialNB_model.fit(train_x, train_y)
	return MultinomialNB_model

def train_Perceptron(train_x, train_y):
	"""
	Training Perceptron network model with train dataset features(train_x) and target(train_y)
	:param train_x:
	:param train_y:
	:return:
	"""

	Perceptron_model = Perceptron(penalty='l2')
	Perceptron_model.fit(train_x, train_y)
	return Perceptron_model


def model_accuracy(trained_model, features, targets):
	"""
	Get the accuracy score of the model
	:param trained_model:
	:param features:
	:param targets:
	:return:
	"""
	accuracy_score = trained_model.score(features, targets)
	return accuracy_score

def predict(logreg, test_feature):
    y_pred = logreg.predict(test_feature)
    y_proba = np.array(logreg.predict_proba(test_feature))
    return y_pred, y_proba

def predict_Perceptron(logreg, test_feature):
    y_pred = logreg.predict(test_feature)
    #y_proba = np.array(logreg.predict_proba(test_feature))
    #return y_pred, y_proba
    return y_pred


def main():
	
	train_file="data/train.csv"
	validation_file= "data/validation.csv"

	orig_file="data/train_original.csv"

	test_file = "data/test.csv"


	train_x, test_x, train_y, test_y = preprocess(train_file, validation_file)

	test_feature = preprocess_test_file(test_file)

	train_x_2,test_x_2=preprocess_bullyingwords(orig_file)



	#  Logistic regression model


	trained_logistic_regression_model = train_logistic_regression(train_x+train_x_2, train_y)
	train_accuracy = model_accuracy(trained_logistic_regression_model, train_x+train_x_2, train_y)
	test_accuracy = model_accuracy(trained_logistic_regression_model, test_x, test_y)
	y_pred, y_proba = predict(trained_logistic_regression_model, test_feature)

	count = 0
	output = './output/output_LR.txt'
	count_file = './output/count_LR.txt'
	out = open(output, 'w+')
	c_file = open(count_file, 'w+')
	
	print("Train Accuracy :: ", train_accuracy)
	print("Test Accuracy :: ", test_accuracy)

	for i in range(len(y_pred)):
		tweet = test_documents[i]
		prediction = y_pred[i]
		if (prediction == 1 or prediction == '1'):
			count += 1
		out.write('%d\t%s\n' % (prediction,tweet))
			
	print(count)
	c_file.write(str(count))


	#MultinomialNB model

	trained_MultinomialNB_model = train_MultinomialNB(train_x+train_x_2, train_y)
	train_MNB_accuracy = model_accuracy(trained_MultinomialNB_model, train_x, train_y)
	test_MNB_accuracy = model_accuracy(trained_MultinomialNB_model, test_x, test_y)

	print("MNB Train Accuracy :: ", train_MNB_accuracy)
	print("MNB Test Accuracy :: ", test_MNB_accuracy)

	y_MNB_pred, y_MNB_proba = predict(trained_MultinomialNB_model, test_feature)

	count = 0
	output_MNB = './output/output_MNB.txt'
	count_file_MNB = './output/count_MNB.txt'
	out = open(output_MNB, 'w+')
	c_file = open(count_file_MNB, 'w+')

	for i in range(len(y_MNB_pred)):
		tweet = test_documents[i]
		prediction = y_MNB_pred[i]
		if (prediction == 1 or prediction == '1'):
			count += 1
		out.write('%d\t%s\n' % (prediction,tweet))

	print(count)
	c_file.write(str(count))

	#Perceptron model

	trained_Perceptron_model = train_Perceptron(train_x+train_x_2, train_y)
	train_Perceptron_accuracy = model_accuracy(trained_Perceptron_model, train_x, train_y)
	test_Perceptron_accuracy = model_accuracy(trained_Perceptron_model, test_x, test_y)

	print("Perceptron Train Accuracy :: ", train_Perceptron_accuracy)
	print("Perceptron Test Accuracy :: ", test_Perceptron_accuracy)

	y_Perceptron_pred = predict_Perceptron(trained_Perceptron_model, test_feature)

	count = 0
	output_Perceptron = './output/output_Perceptron.txt'
	count_file_Perceptron = './output/count_Perceptron.txt'
	out = open(output_Perceptron, 'w+')
	c_file = open(count_file_Perceptron, 'w+')

	for i in range(len(y_Perceptron_pred)):
		tweet = test_documents[i]
		prediction = y_Perceptron_pred[i]
		if (prediction == 1 or prediction == '1'):
			count += 1
		out.write('%d\t%s\n' % (prediction,tweet))

	print(count)
	c_file.write(str(count))

	
if __name__ == "__main__":
	main()
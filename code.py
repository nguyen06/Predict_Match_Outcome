

import csv
import sys
import operator
import matplotlib.pyplot as plt
from functools import reduce
from sklearn.naive_bayes import GaussianNB,MultinomialNB, BernoulliNB
from sklearn import svm
from sklearn.linear_model import Perceptron,LogisticRegression
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier

#from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import neighbors

#from sklearn.neural_network import MLPClassifier

"""
#if we give two team nam, if will go to your_list and find the team data
team01_name = input("please enter home team: ")
team02_name = input("please enter away team: ")
team_list = [team01_name,team02_name]
"""
#create a list of all pair in 2016 matches, to test the accuracy for model
with open("predict.csv",'r') as t:
	team_reader = csv.reader(t)
	team_pair_list = list(team_reader)
	#print(team_pair_list)
#create two list, one for actual, one for predict, then compare them for accuracy
actual_label_list = []
predict_label_list = []

for pair in team_pair_list:
	actual_label_list.append(pair[2])

#print(team_pair_list)
"""
with open('data01.csv', 'r') as f:

		#['Date', 'HomeTeam', 'HS', 'HST', 'HF', 'HC', 'HY', 'HR', 'AwayTeam', 'AS', 'AST', 'AF', 'AC', 'AY', 'AR', 'FTR']
		reader = csv.reader(f)
		your_list = list(reader)
"""
"""
tr = []
labe =[]
for line in your_list:
	elem = [line[1],line[2],line[3],line[4],line[5],line[6],line[8],line[9],line[10],line[11],line[12],line[13]]
	tr.append(elem)
	lab = line[14]
	labe.append(lab)
tr = np.array(tr).astype(np.int)
#print(labe)
"""
for team_list in team_pair_list:
	
	#we generate the list of lines, which also the list from the csv data
	with open('data01.csv', 'r') as f:

		#['Date', 'HomeTeam', 'HS', 'HST', 'HF', 'HC', 'HY', 'HR', 'AwayTeam', 'AS', 'AST', 'AF', 'AC', 'AY', 'AR', 'FTR']
		reader = csv.reader(f)
		your_list = list(reader)
		
	tr = []
	labe =[]
	for line in your_list:
		elem = [line[1],line[2],line[3],line[4],line[5],line[6],line[8],line[9],line[10],line[11],line[12],line[13]]
		tr.append(elem)
		lab = line[14]
		labe.append(lab)
	tr = np.array(tr).astype(np.int)




	"""
	
	#Now we go and find the given team data, After finding the data we need, we generate them into traning data
	
	data_train = []
	label_train = []
	#print(team_list[0] + " " + team_list[1])
	for line in your_list:
		#print(line[0] +" "+line[7] + " "+team_list[0] + " " +team_list[1])
		if line[0] == team_list[0] and line[7] == team_list[1] or line[7] == team_list[0] and line[0] == team_list[1]:
			print(line[0] +","+ line[1] +","+line[2]+","+line[3])
			elem = [line[1],line[2],line[3],line[4],line[5],line[6],line[8],line[9],line[10],line[11],line[12],line[13]]
			data_train.append(elem)
			la = line[14]
			label_train.append(la)
	# Since each element of the training data now is string, we need to make them into int inoder to run prdiction
	data_train = np.array(data_train).astype(np.int)
	#print(label_train)
	#print(data_train)
	"""
	"""
	Now, we need to calculate how well the team work in the last previous years
	"""
	class Dictlist(dict):
	    def __setitem__(self, key, value):
	        try:
	            self[key]
	        except KeyError:
	            super(Dictlist, self).__setitem__(key, [])
	        self[key].append(value)

	team01_data = Dictlist()
	team02_data = Dictlist()

	#find the name of given team, and generate all data belong to it
	for line in your_list:
		if line[0] == team_list[0]:
			team01 = [line[1],line[2],line[3],line[4],line[5],line[6]]
			team01_data[team_list[0]] = team01
		elif line[7] == team_list[0]:
			team011 = [line[8],line[9],line[10],line[11],line[12],line[13]]
			team01_data[team_list[0]] = team011
			
		if line[0] == team_list[1]:
			team02 = [line[1],line[2],line[3],line[4],line[5],line[6]]
			team02_data[team_list[1]] = team02
		elif line[7] == team_list[1]:
			team022 = [line[8],line[9],line[10],line[11],line[12],line[13]]
			team02_data[team_list[1]] = team022

	#print to check if there are correct data
	#print(team01_data)
	#print(team02_data)

	test_for_predict =[] # test data is calculate from average performance of two given team
	#now calcuate how well the team perform for each attribute from pervious yes for team 01, and append 
	#into the test list

	for key, value in team01_data.items():
		#this list has element type string, we conver to int
		value = np.array(value).astype(np.int)
		team01_atts = []
		#there are 6 attributes, we need to go through each attribute and put them into the list
		for i in range(6):
			#call the average, which we use to calculate the average for each attribute
			ave = 0
			list_of_att = []
			#go through the list of value
			for list_att in value:
				list_of_att.append(list_att[i])
			#Calculate the average foe each
			ave = sum(list_of_att)/len(list_of_att)
			team01_atts.append(ave)
		test_for_predict.append(team01_atts)

	# do the same thing with team 2
	for key, value in team02_data.items():
		#this list has element type string, we conver to int
		value = np.array(value).astype(np.int)
		team02_atts = []
		#there are 6 attributes, we need to go through each attribute and put them into the list
		for i in range(6):
			#call the average, which we use to calculate the average for each attribute
			ave = 0
			list_of_att = []
			#go through the list of value
			for list_att in value:
				list_of_att.append(list_att[i])
			#Calculate the average foe each
			ave = sum(list_of_att)/len(list_of_att)
			team02_atts.append(ave)
		test_for_predict.append(team02_atts)
	# from two list in one list, compute into one list
	test_for_predict = reduce(lambda x,y:x+y,test_for_predict)

	#calcualte to int for test_for_predict
	test_for_predict = np.array(test_for_predict).astype(np.int)
	
	print(test_for_predict)
	#n_neighbors = 15

	clf = GaussianNB() #42
	#clf = svm.SVC(kernel='rbf') #40 regression
	#clf = LogisticRegression() #42
	#clf = MultinomialNB() #38
	#clf = BernoulliNB() #37
	#clf = DecisionTreeClassifier() # 36
	#clf = Perceptron() #37
	#clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=1000,learning_rate=0.4, random_state=42) #38
	#clf = SGDClassifier(loss="hinge", penalty="l2") #43%
	#clf = neighbors.KNeighborsClassifier(n_neighbors) #37
	#clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)


	clf = clf.fit(tr,labe)
	#clf = clf.fit(data_train,label_train)
	predict = clf.predict(test_for_predict)
	label = 'a'
	for i in predict:
		label = i
	predict_label_list.append(label)
	#print(type(pre))
	#update the CSV data for new test
	#print(team01_name)
	row = []
	row.append(team_list[0])
	row.append(test_for_predict[0])
	row.append(test_for_predict[1])
	row.append(test_for_predict[2])
	row.append(test_for_predict[3])
	row.append(test_for_predict[4])
	row.append(test_for_predict[5]) #,test_for_predict[2],test_for_predict[3],test_for_predict[4],test_for_predict[5])
	row.append(team_list[1])
	row.append(test_for_predict[6])
	row.append(test_for_predict[7])
	row.append(test_for_predict[8])
	row.append(test_for_predict[9])
	row.append(test_for_predict[10])
	row.append(test_for_predict[11])
	#row.append(test_for_predict[6],test_for_predict[7],test_for_predict[8],test_for_predict[9],test_for_predict[10],test_for_predict[11])
	row.append(label)
	print(row)


	with open(r'data01.csv', 'a') as f:
	                writer = csv.writer(f)
	                writer.writerow(row)


# compare two list, two see how many pair they are them same
count = 0
for i in range(100):
	if actual_label_list[i] == predict_label_list[i]:
		count = count + 1
print(count)






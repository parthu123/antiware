import random
import math
import csv
import numpy
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
import random

def loadCsv(filename):
	y=[]
	d=[]
	lines = csv.reader(open(filename, "rb"))
	dataset = list(lines)
	dataset = dataset[1:]
	#print "len:",len(dataset)
	test=[]
	actualLabel=[]
	for i in range(0,60):
		num=random.randint(0,len(dataset)-1)
		#print "num:",num
		test.append(dataset[num])
		#print "t:",test
		actualLabel.append(int(dataset[num][-1]))
		del test[i][-1]
		dataset.pop(num)
	
	for row in dataset:
		i=0
	for i in range(len(dataset)):
		dataset[i] = [int(x) for x in dataset[i]]
		
		y.append(dataset[i][-1])
		del dataset[i][-1]
		d.append(numpy.array(dataset[i]))
	d=numpy.array(d)
	return d,y,test,actualLabel
	
	
	
filename='for_meta.csv'
x,y,test,actualLabel=loadCsv(filename)
a=[]
clf=svm.SVC(kernel='linear', C = 1.0)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1) #0.1,1
#clf.fit(X_train,y_train)
clf.fit(x,y)
print "Sc:",clf.score(X_test, y_test)
#print "Act:",actualLabel
'''print
print "Analysis"
print
print "Predicted class\t\tActual class"
print'''
'''true=0
false=0
for i in range(0,10):
	p=clf.predict([test[i]])
	if p[0]==1: #1
		pred= "B"
	else:
		pred="M"
		
	if actualLabel[i]==0:#0
		act="B"
	else:
		act="M"
	#print "\t",pred,"\t\t\t",act
	#print
	if actualLabel[i]==p[0]:
		true+=1
	else:
		false+=1
	
print "True predictions:",true
print "False predictions:",false
res = true+false	
#tpr = float(true)/float(res)
#fpr = float(false)/float(res)
#print "TruePositiveRate:",tpr
#print "FalsePositiveRate:",fpr
'''
TP = 0
FP = 0
TN = 0
FN = 0
#k = cohen_kappa_score(X_test, y_test,labels=None, weights=None)
for i in range(0,60):
        p=clf.predict([test[i]])

	#print p[0]
	if p[0]==actualLabel[i] and p[0]==1:
	        TP += 1
	
	elif p[0]==actualLabel[i] and p[0]==0:
	        TN += 1
	
	elif p[0]!=actualLabel[i] and p[0]==1:
		FP += 1
	else:
		FN +=1

		
	
#print "True predictions:",true
#print "False predictions:",false
#res = true+false	
#tpr = float(true)/float(res)
#print TP,TN,FP,FN
tpr = float(TP) / float(TP+FN)
fnr = float(FN)/float(FN+TP)
fpr = float(FP)/float(TN+FP)
tnr = float(TN) / float(TN+FP)
tpr = float(TP) / float(TP+FN)
fnr = float(FN)/float(FN+TP)
fpr = float(FP)/float(TN+FP)
tnr = float(TN) / float(TN+FP)
first = (float(TP)+float(TN))/(float(TP)+float(TN)+float(FP)+float(FN))
second = (float(FP)+float(FN))/(float(TP)+float(TN)+float(FP)+float(FN))
third = (float(TP)+float(FP))/(float(TP)+float(TN)+float(FP)+float(FN))
fourth = (float(TN)+float(FN))/(float(TP)+float(TN)+float(FP)+float(FN))
last = (float(TP)+float(FN))/(float(TP)+float(TN)+float(FP)+float(FN))
prob_yes = first * third
prob_no = second * fourth
final = prob_yes + prob_no
kappa = (last - final)/(1-final)
print "TruePositiveRate:",tpr
print "FalseNegativeRate:",fnr
print "FalsePositiveRate:",fpr
print "TrueNegativeRate:",tnr
print "kappa score:",kappa


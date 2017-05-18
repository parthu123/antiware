from __future__ import division
import csv 
import random
import math
import collections,itertools
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import svm
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier








#step 1 : Load the data
def LoadData(filename):
        file = open(filename,'r')
        lines = csv.reader(file)
        dataset = list(lines)
        return dataset




#step 2: Create classlabls
def assignClassLabels(data):
        list_for_labels = [] 
        val = 0
        retrive_class = [row[-1] for row in data]
        for j in range(1,len(retrive_class)):
                if(int(retrive_class[j]) == 0):
                        list_for_labels.append(0)
                else :
                        list_for_labels.append(1)
        return list_for_labels
     
  



#step 3: Calculating probabilities for classlabels
def CalProbabilities(list_one):
        true_count = 0
        false_count = 0
        for item in list_one:
                if(item == 0):
                        true_count = true_count + 1
                else :
                        false_count = false_count + 1
        true_prob = true_count/len(list_one)
        false_prob = false_count/len(list_one)
        return true_prob,false_prob





#step 4:Calculating Entropy for the classlabels
def CalClassEntropy(true,false):
        if(true != 0 and false != 0):
          info = true * math.log(true)/math.log(2) + false * math.log(false)/math.log(2)
        elif(true == 0):
          info = false * (math.log(false)/math.log(2))
        else :
          info = true * (math.log(true)/math.log(2))
        return 0-info
                
        
                
                        
#step 5:calculate  probabliteis for each attribute in a column
def CalProbForAttr(data,list_of_labels,i):
          dict_for_attr = {}
          count = 0
          formality=0
          list_of_gains = list()
          sum_of_entr = 0
          retrive_attr = [row[0] for row in data]
          #print(retrive_attr)
           #chnage thi to row[i]
          '''for item in retrive_attr:
               if(formality == 0):
                    formality = 1
                    continue
               if(item not in dict_for_attr):
                    dict_for_attr[item] = []
               else :
                    dict_for_attr[item] = dict_for_attr[item] + 1
          print dict_for_attri]'''
          for row in data:
            if(row[i] not in dict_for_attr):
              l = list()
              l.append(row)
              dict_for_attr[row[i]] = l
            else :
              l = dict_for_attr[row[i]]
              l.append(row)
              dict_for_attr[row[i]] = l
          #print(len(dict_for_attr["APP_BOOKS_AND_REFERENCE_CATEGORY"]))
           

        #Calculating probabilities for each attribute in the column
          list_of_prob = list()
          for key in dict_for_attr :
                value = dict_for_attr[key]
                prob = len(value)/len(data)
                list_of_prob.append(prob)
                true , false = CalProbabilities(value)
                entropy_formula = (len(value)/len(data))* CalClassEntropy(true/len(value),false/len(value))
                sum_of_entr += entropy_formula
          return dict_for_attr ,sum_of_entr
     


#step 6:Now,we calculate information for the column 
def CalInfoForColumn(dict_of_val):
    #see_this_once = 0
    info_of_attr = 0
    for i in range(len(dict_of_val)):
        info_of_attr = info_of_attr + ((dict_of_val.keys()[i]/len(data)) * CalClassEntropy(dict_of_val.values()[i][0]/dict_of_val.keys()[i],dict_of_val.values()[i][1]/dict_of_val.keys()[i]))
    return info_of_attr



#step 7 : Applying svm to our new extracted features
#def applyClassifier():
  #training_data = np.loadtxt('outtwo.csv',delimiter = ',')
  #x = training_data[1:98]
  #for i in range(len(x)):
  #  x[i] = int(x[i])
  #print x
  #print len(x)

  #y = training_data[-1]
  #y = training_data[:,1:]
  #clf = ExtraTreesClassifier(n_estimators=100).fit(x, y)
  #data_test = np.loadtxt('outtwo.csv', delimiter=',')
  #print(clf.predict(data_test))
















        
   
#Calculating each tuple occurence with the count and Returning this as a dictionary along with values
'''
          dict_for_cal_prob = {}
          for i in range(len(dict_for_attr)) :
                attr_true_count = 0
                attr_false_count = 0
                for j in range( dict_for_attr.values()[i]):
                         if(list_of_labels[j] == "True"):
                                attr_true_count = attr_true_count + 1
                         else:
                                attr_false_count = attr_false_count + 1
                if(dict_for_attr.values()[i] not in dict_for_cal_prob):
                    dict_for_cal_prob.update({dict_for_attr.values()[i]:[attr_true_count,attr_false_count]})
                #if(any(k == 0 for k in dict_for_cal_prob.itervalues())):
                    #continue
          print(dict_for_cal_prob)
          return dict_for_cal_prob
'''





#step 8: 
#def svmClassifier(actual_data,extracte 
#Here is the Main function
if __name__ == "__main__" :

        file_name = 'only_per.csv'
        df = pd.read_csv(file_name)
        # print(df)
        data = LoadData(file_name)
        #print(data)
        list_of_labels = assignClassLabels(data)
        #Listing probabilities of malware(true)/benign(false) class labels
        true, false = CalProbabilities(list_of_labels)
        #calculating entropy of classlabels
        info_val = CalClassEntropy(true/(true+false),false/(true+false))
        #print(info_val)
        list_for_gain = list()
        dict_for_gain = {}
        for i in range(len(data[0])) :
          if(data[0][i] == "MALWARE_DETECTION_COUNT"):
            continue
          dict_for_fea ,var = CalProbForAttr(data,list_of_labels,i)
          #print(dict_for_fea.keys())
          gain_val = info_val - var
          list_for_gain.append(gain_val)
          dict_for_gain.update({data[0][i]:gain_val})
        check_val = sorted(list_for_gain)
        see_that_list = list()
        svm_list = list()

        for j in range(100):
          see_that_list.append(check_val[j]) 
        for k in range(len(dict_for_gain)):
          if(dict_for_gain.values()[k] in see_that_list):
            svm_list.append(dict_for_gain.keys()[k])
        create_newone = df[svm_list]
        #applyClassifier()
        #print(create_newone.shape)
        create_newone.to_csv("ig_per_fs.csv")
        












'''        dict_for_svm = {label:[] for label in svm_list}
        # Create a new reverse lookup dictionary
        rev_dict = {}
        rd = csv.reader(open(file_name, 'r'))
        feature_labels = rd.next()
        for i in range(len(feature_labels)):
          if feature_labels[i] in svm_list:
            rev_dict[i] = feature_labels[i]
        # Take each tuple at a time, and add appropriately
        #print(rev_dict)
        for tup in rd:
          for i in range(len(data[0])):
            if rev_dict.has_key(i):
              dict_for_svm[rev_dict[i]].append(tup[i])
        print(dict_for_svm)
        # Check
        #for key in dict_for_svm.keys():
           #print key, len(dict_for_svm[key])
'''
















              

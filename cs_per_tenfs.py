from __future__ import division
import sys
import csv
import math
import random
import numpy
import pandas as pd
from scipy.stats import chisquare



#step 1: Loading Our dataset
def LoadDataset(filename):
    fileName = open(filename,'rb')
    lines = csv.reader(fileName)
    dataset = list(lines)
    return dataset

#step 2:Calculating Occurence of each tuple
def CalCount(dataset):
            formality = 0
            n = 5000
            list_for_cs = list()
            for column in range(len(dataset[0])-1) :
                if(formality == 0) :

                    formality = 1
                    continue
                first_one = second_one = third_one = fourth_one = 0
                for row in range(1,len(dataset)-1):
                    #print(dataset[column][-1])
                    num = int(dataset[row][column])
                    label = int(dataset[row][-1])
                    if(num == 0 and label == 0):
                        first_one = first_one + 1
                    elif(num == 0 and label > 0):
                        second_one = second_one + 1
                    elif(num == 1 and label == 0):
                        third_one = third_one + 1
                    elif(num == 1 and label > 0):
                        fourth_one = fourth_one + 1
                #print(first_one,second_one,third_one,fourth_one)
                if(third_one == 0 and fourth_one == 0):
                    chi_square = 0
                    list_for_cs.append(chi_square)
                else :
                    chi_square = ((n * n) * ((second_one * third_one) - (first_one * fourth_one))) / ((first_one + second_one) * (fourth_one + third_one) * (second_one + fourth_one) * (first_one + third_one)) 
                    chi_square = abs(chi_square)
                    list_for_cs.append(chi_square)
        
            return list_for_cs
            











#Here is our main Function
if __name__ == "__main__" :
     file_name = 'only_for_cs.csv'
     data = LoadDataset(file_name)
     df = pd.read_csv(file_name)
     #print(len(data[0]))
     list_of_cs = CalCount(data)
     dict_for_cs = {}
     for i in range(0,849):
        if(data[0][i] == "MALWARE_DETECTION_COUNT"):
            break
        dict_for_cs.update({data[0][i]:list_of_cs[i]})
     #print(dict_for_cs)
     check_val = sorted(list_of_cs,reverse=True)
     #print(check_val)
     see_that_list = list()
     cs_list = list()
     for j in range(10): # try with different values
        see_that_list.append(check_val[j]) 
     #print(len(see_that_list))
     for k in range(len(dict_for_cs)):
        if(dict_for_cs.values()[k] in see_that_list):
            cs_list.append(dict_for_cs.keys()[k])
     create_newone = df[cs_list]
        #print(create_newone.shape)
     create_newone.to_csv("cs_per_tenfs.csv")







#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 11:05:59 2022

@author: patrikchrenko
"""

import os
import csv

myList = []

keyDict = {
    "True" : 1,
    "False" : 0,
    "NA" : 0,
    "Grvl" : 1,
    "Pave" : 2,
    "A" : 1,
    "C" : 2,
    "FV" : 3,
    "I" : 4,
    "RH" : 5,
    "RL" : 6,
    "RP" : 7,
    "RM" : 8,
    
    }

with open('../Original/train.csv', 'r') as file:
    myFile = csv.reader(file)
    for row in myFile:
        myList.append(row)
        
    
    
    # for i in range (10):
    #     for n in range (len(myList[6])):
    #         if myList[i][n] in keyDict:
    #             print(myList[i][n])
    #             myList[i][n] = keyDict.get(myList[i][n])
                
    # for i in range (len(myList)):
    #     for n in range (len(myList[0])):
    #         if myList[i][n] == null:
    #             print(myList[i][n])
                
    for i in range(len(myList)):
        for n in range (len(myList[0])):
            if myList[i][n] == "NA":
                myList[i][n] = 0
                

writer = csv.writer(open("../Laundried/nullCleaned.csv", 'w'))
for row in myList:
    writer.writerow(row)

    
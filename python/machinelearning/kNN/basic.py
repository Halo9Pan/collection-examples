'''
Created on Sep 16, 2010
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)
            
Output:     the most popular class label

@author: pbharrin
'''
from os import listdir
from numpy import *
import operator
import util

def create_data_set():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def file_matrix(filename):
    love_dictionary={'largeDoses':3, 'smallDoses':2, 'didntLike':1}
    fr = open(filename)
    array_lines = fr.readlines()
    lines_number = len(array_lines)            #get the number of lines in the file
    return_mat = zeros((lines_number,3))       #prepare matrix to return
    class_label_vector = []                    #prepare labels return   
    index = 0
    for line in array_lines:
        line = line.strip()
        list_from_line = line.split('\t')
        return_mat[index,:] = list_from_line[0:3]
        if(list_from_line[-1].isdigit()):
            class_label_vector.append(int(list_from_line[-1]))
        else:
            class_label_vector.append(love_dictionary.get(list_from_line[-1]))
        index += 1
    return return_mat,class_label_vector
    
def datingClassTest():
    ho_ratio = 0.50      #hold out 10%
    dating_data_mat,dating_labels = file_matrix('datingTestSet2.txt')       #load data setfrom file
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    m = norm_mat.shape[0]
    num_test_vecs = int(m*ho_ratio)
    error_count = 0.0
    for i in range(num_test_vecs):
        classifier_result = classify(norm_mat[i,:],norm_mat[num_test_vecs:m,:],dating_labels[num_test_vecs:m],3)
        print("the classifier came back with: %d, the real answer is: %d" %(classifier_result, dating_labels[i]))
        if (classifier_result != dating_labels[i]): error_count += 1.0
    print("the total error rate is: %f" %(error_count/float(num_test_vecs)))
    print(error_count)
    
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percent_tats = float(raw_input(\
                                  "percentage of time spent playing video games?"))
    ffMiles = float(raw_input("frequent flier miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    dating_data_mat, dating_labels = file_matrix('datingTestSet2.txt')
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    inArr = array([ffMiles, percent_tats, iceCream, ])
    classifier_result = classify((inArr - \
                                  min_vals)/ranges, norm_mat, dating_labels, 3)
    print("You will probably like this person: %s" %(resultList[classifier_result - 1]))
    
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')           #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')        #iterate through the test set
    error_count = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifier_result = classify(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" %(classifier_result, classNumStr))
        if (classifier_result != classNumStr): error_count += 1.0
    print("\nthe total number of errors is: %d" %(error_count))
    print("\nthe total error rate is: %f" %(error_count/float(mTest)))


# coding: utf-8

# In[ ]:

import sys
import string
import copy
import math
import numpy as np
import random
from random import shuffle
from collections import Counter
from operator import itemgetter


def process_str(s):
    rem_punc = str.maketrans('', '', string.punctuation)
    return s.translate(rem_punc).lower().split()

def read_dataset(file_name):
    dataset = []
    with open(file_name) as f:
        for line in f:
            index, class_label, text = line.strip().split('\t')
            words = process_str(text)
            dataset.append( (int(class_label), words) )

    return dataset

def get_most_commons(dataset, skip=100, total=1000):
    counter = Counter()
    for item in dataset:
        counter = counter + Counter(set(item[1]))

    temp = counter.most_common(total+skip)[skip:]
    words = [item[0] for item in temp]
    return words

def generate_vectors(dataset, common_words):
    d = {}
    for i in range(len(common_words)):
        d[common_words[i]] = i
    vectors = []
    labels = []
    for item in dataset:
        vector = [0] * len(common_words)
        for word in item[1]:
            if word in d:
                vector[d[word]] = 1
        vectors.append(vector)
        labels.append(item[0])
    return np.array(vectors), np.array(labels)
##########################CrossValidation function is for later analysis#############################
def CrossValidation(dataSetX, dataSetY, k,TSS):
    Allset = []
    allindex = list(range(2000))
    kfoldindex = []
    for i in range(k):
        random.shuffle(allindex)
        eachindex = allindex[:200]
        allindex = [item for item in allindex if item not in eachindex]
        kfoldindex.append(eachindex)
    for eachkfoldindex in kfoldindex:
        testsetx = dataSetX[eachkfoldindex]
        testsety = dataSetY[eachkfoldindex]
        trainindex = [item for item in list(range(2000)) if item not in eachkfoldindex]
        trainindexReal = random.sample(trainindex,TSS)
        trainsetx = dataSetX[trainindexReal]
        trainsety = datasetY[trainindexReal]
        eachset = [[testsetx,testsety],[trainsetx,trainsety]]
        Allset.append(eachset)
    return Allset


def Gini(classlabelset):
    p_ispositive = np.count_nonzero(classlabelset) / classlabelset.size
    p_isnegative = 1 - p_ispositive
    gini = 1 - (p_ispositive * p_ispositive + p_isnegative * p_isnegative)
    return gini

def testsplit(index, dataSetX, dataSetY):
    subset0X = dataSetX[dataSetX[:,index]==0]
    subset0Y = dataSetY[dataSetX[:,index]==0]
    subset1X = dataSetX[dataSetX[:,index]==1]
    subset1Y = dataSetY[dataSetX[:,index]==1]
    left = subset0X,subset0Y
    right = subset1X,subset1Y
    return left,right

def ChooseBestFeatureToSplit(dataSetX, dataSetY):
    global bestFeature
    baseGini = Gini(dataSetY)
    bestGiniGain = 0.0
    for i in range(len(dataSetX[0])):  # i = 1000
        subset0X = dataSetX[dataSetX[:, i] == 0]
        subset0Y = dataSetY[dataSetX[:, i] == 0]
        subset1X = dataSetX[dataSetX[:, i] == 1]
        subset1Y = dataSetY[dataSetX[:, i] == 1]
        if subset0Y.size == 0:
            GiniGain = baseGini - (Gini(subset1Y) * subset1Y.size / dataSetY.size)
        elif subset1Y.size == 0:
            GiniGain = baseGini - (Gini(subset0Y) * subset0Y.size / dataSetY.size)
        else:
            GiniGain = baseGini - (
            Gini(subset0Y) * subset0Y.size / dataSetY.size + Gini(subset1Y) * subset1Y.size / dataSetY.size)
        if GiniGain > bestGiniGain:
            bestGiniGain = GiniGain
            bestFeature = i
    return bestFeature

def GetSplit(dataSetX, dataSetY):
    index = ChooseBestFeatureToSplit(dataSetX, dataSetY)
    left, right = testsplit(index, dataSetX, dataSetY)
    return {'index': index, 'left': left, 'right': right}

def TerminalMajorityCount(labels):
    if sum(labels) >= 0.5 * len(labels):
        return 1
    else:
        return 0

def split(node, max_depth, min_size, depth):
    left = node['left']
    right = node['right']
    if len(left[0]) == 0 and len(right[0]) != 0:
        node['left'] = TerminalMajorityCount(right[1])
        node['right'] = TerminalMajorityCount(right[1])
    if len(left[0]) != 0 and len(right[0]) == 0:
        node['left'] = TerminalMajorityCount(left[1])
        node['right'] = TerminalMajorityCount(left[1])
        return
    if depth >= max_depth:
        node['right'], node['left'] = TerminalMajorityCount(right[1]), TerminalMajorityCount(left[1])
        return
    if len(left[0]) <= 10:
        node['left'] = TerminalMajorityCount(left[1])
        if len(right[0]) <= 10:
            node['right'] = TerminalMajorityCount(right[1])
            return
        else:
            node['right'] = GetSplit(right[0], right[1])

            split(node['right'], max_depth, min_size, depth + 1)
    else:
        node['left'] = GetSplit(left[0], left[1])

        split(node['left'], max_depth, min_size, depth + 1)
        if len(right[0]) <= 10:
            node['right'] = TerminalMajorityCount(right[1])
            return
        else:
            node['right'] = GetSplit(right[0], right[1])

            split(node['right'], max_depth, min_size, depth + 1)

def build_tree(trainX, trainY, max_depth, min_size):
    root = GetSplit(trainX, trainY)
    split(root, max_depth, min_size, 1)
    return root

def PredictionOfOne(node,x):
    if x[node['index']]==1:
        if isinstance(node['right'], dict):
            return PredictionOfOne(node['right'], x)
        else:
            return node['right']
    else:
        if isinstance(node['left'], dict):
            return PredictionOfOne(node['left'], x)
        else:
            return node['left']

def TestDT(trainX,trainY,testX,testY):
    mytree = build_tree(trainX, trainY,10,10)
    predictions = []
    count = 0
    for row in testX:
        prediction = PredictionOfOne(mytree, row)
        predictions.append(prediction)
    for i in range(len(predictions)):
        if predictions[i] !=  testY[i]:
            count +=1
    loss = count / len(testY)
    return loss

#################################Bagged Decision Tree################################

def BAG_DT(trainX,trainY,max_depth, min_size):
    trees = []
    for i in range(50):
        trainX_BGDT=np.zeros(1000,dtype = int)
        trainY_bgdt=[]
        for i in range(len(trainY)):
            idx = np.random.randint(len(trainY))
            trainX_BGDT_one= trainX[idx,:]
            trainY_bgdt_one = trainY[idx]
            trainX_BGDT =  np.vstack([trainX_BGDT, trainX_BGDT_one])
            trainY_bgdt.append(trainY_bgdt_one)
        trainX_BGDT = np.delete(trainX_BGDT, (0), axis=0)
        trainY_BGDT = np.asarray(trainY_bgdt)
        tree = build_tree(trainX_BGDT, trainY_BGDT,10,10)
        trees.append(tree)
    return trees

def Test_BAG_DT(trainX, trainY, testX, testY):
    BAG_Tree = BAG_DT(trainX, trainY, 10, 10)
    pred_all = []
    for eachtree in BAG_Tree:
        pred_all_for_one_tree = []
        for row in testX:
            pred_of_one_tree = PredictionOfOne(eachtree, row)
            pred_all_for_one_tree.append(pred_of_one_tree)
        pred_all.append(pred_all_for_one_tree)

    pred_all_array = np.asarray(pred_all).T
    result = []
    for i in pred_all_array:
        result.append(TerminalMajorityCount(i))
    count = 0
    for i in range(len(result)):
        if result[i] != testY[i]:
            count += 1
    loss = count / len(testY)
    return loss

##################Random Forest######################################

def ChooseBestFeatureToSplit_RF(dataSetX, dataSetY):
    global bestFeature
    Ridx = np.random.randint(0, 1000, size=(31))
    baseGini = Gini(dataSetY)
    bestGiniGain = 0.0
    for idx in Ridx:
        subset0X = dataSetX[dataSetX[:, idx] == 0]
        subset0Y = dataSetY[dataSetX[:, idx] == 0]
        subset1X = dataSetX[dataSetX[:, idx] == 1]
        subset1Y = dataSetY[dataSetX[:, idx] == 1]
        if subset0Y.size == 0:
            GiniGain = baseGini - (Gini(subset1Y) * subset1Y.size / dataSetY.size)
        elif subset1Y.size == 0:
            GiniGain = baseGini - (Gini(subset0Y) * subset0Y.size / dataSetY.size)
        else:
            GiniGain = baseGini - (
            Gini(subset0Y) * subset0Y.size / dataSetY.size + Gini(subset1Y) * subset1Y.size / dataSetY.size)
        if GiniGain > bestGiniGain:
            bestGiniGain = GiniGain
            bestFeature = idx
    return bestFeature

def GetSplit_RF(dataSetX, dataSetY):
    index = ChooseBestFeatureToSplit_RF(dataSetX, dataSetY)
    left, right = testsplit(index, dataSetX, dataSetY)
    return {'index': index, 'left': left, 'right': right}

def split_RF(node, max_depth, min_size, depth):
    left = node['left']
    right = node['right']
    if len(left[0]) == 0 and len(right[0]) != 0:
        node['left'] = TerminalMajorityCount(right[1])
        node['right'] = TerminalMajorityCount(right[1])
    if len(left[0]) != 0 and len(right[0]) == 0:
        node['left'] = TerminalMajorityCount(left[1])
        node['right'] = TerminalMajorityCount(left[1])
        return
    if depth >= max_depth:
        node['right'], node['left'] = TerminalMajorityCount(right[1]), TerminalMajorityCount(left[1])
        return
    if len(left[0]) <= 10:
        node['left'] = TerminalMajorityCount(left[1])
        if len(right[0]) <= 10:
            node['right'] = TerminalMajorityCount(right[1])
            return
        else:
            node['right'] = GetSplit_RF(right[0], right[1])
            split(node['right'], max_depth, min_size, depth + 1)
    else:
        node['left'] = GetSplit_RF(left[0], left[1])

        split(node['left'], max_depth, min_size, depth + 1)
        if len(right[0]) <= 10:
            node['right'] = TerminalMajorityCount(right[1])
            return
        else:
            node['right'] = GetSplit_RF(right[0], right[1])
            split(node['right'], max_depth, min_size, depth + 1)


def build_tree_RF(trainX, trainY, max_depth, min_size):
    root = GetSplit_RF(trainX, trainY)
    split_RF(root, max_depth, min_size, 1)
    return root


def RandomForest(trainX,trainY,max_depth, min_size):
    trees = []
    for i in range(50):
        trainX_RF=np.zeros(1000,dtype = int)
        trainY_rf=[]
        for i in range(len(trainY)):
            idx = np.random.randint(len(trainY))
            trainX_RF_one = trainX[idx,:]
            trainY_rf_one = trainY[idx]
            trainX_RF =  np.vstack([trainX_RF, trainX_RF_one])
            trainY_rf.append(trainY_rf_one)
        trainX_RF = np.delete(trainX_RF, (0), axis=0)
        trainY_RF = np.asarray(trainY_rf)
        tree = build_tree(trainX_RF, trainY_RF,10,10)
        trees.append(tree)
    return trees

def Test_RF(trainX, trainY, testX, testY):
    RF_Tree = RandomForest(trainX, trainY, 10, 10)
    pred_all = []
    for eachtree in RF_Tree:
        pred_all_for_one_tree = []
        for row in testX:
            pred_of_one_tree = PredictionOfOne(eachtree, row)
            pred_all_for_one_tree.append(pred_of_one_tree)
        pred_all.append(pred_all_for_one_tree)

    pred_all_array = np.asarray(pred_all).T
    result = []
    for i in pred_all_array:
        result.append(TerminalMajorityCount(i))
    count = 0
    for i in range(len(result)):
        if result[i] != testY[i]:
            count += 1
    loss = count / len(testY)
    return loss

####################Boosting Decison Tree###########################

def WeightData(dataSetX, dataSetY, weight):
    dataSetX_w = dataSetX * weight
    dataSetY_w = dataSety * weight
    return dataSetX_w, dataSetY_w

def Gini_BST(classlabelset, weight):
    p_ispositive_weighted = sum(weight[classlabelset == 1]) / sum(weight)
    p_isnegative_weighted = 1 - p_ispositive_weighted
    gini = 1 - (p_ispositive_weighted * p_ispositive_weighted + p_isnegative_weighted * p_isnegative_weighted)
    return gini

def testsplit_BST(index, dataSetX, dataSetY, weight):
    subset0X = dataSetX[dataSetX[:, index] == 0]
    subset0Y = dataSetY[dataSetX[:, index] == 0]
    subset1X = dataSetX[dataSetX[:, index] == 1]
    subset1Y = dataSetY[dataSetX[:, index] == 1]
    weight0 = weight[dataSetX[:, index] == 0]
    weight1 = weight[dataSetX[:, index] == 1]
    left = subset0X, subset0Y, weight0
    right = subset1X, subset1Y, weight1
    return left, right

def ChooseBestFeatureToSplit_BST(dataSetX, dataSetY, weight):
    global bestFeature
    baseGini = Gini_BST(dataSetY, weight)
    bestGiniGain = 0.0
    for i in range(len(dataSetX[0])):  # i = 1000
        subset0X = dataSetX[dataSetX[:, i] == 0]
        subset0Y = dataSetY[dataSetX[:, i] == 0]
        subset1X = dataSetX[dataSetX[:, i] == 1]
        subset1Y = dataSetY[dataSetX[:, i] == 1]
        weight0 = weight[dataSetX[:, i] == 0]
        weight1 = weight[dataSetX[:, i] == 1]
        if subset0Y.size == 0:
            GiniGain = baseGini - (Gini_BST(subset1Y, weight1) * subset1Y.size / dataSetY.size)
        elif subset1Y.size == 0:
            GiniGain = baseGini - (Gini_BST(subset0Y, weight0) * subset0Y.size / dataSetY.size)
        else:
            GiniGain = baseGini - (Gini_BST(subset0Y, weight0) * subset0Y.size / dataSetY.size + Gini_BST(subset1Y,
                                                                                                          weight1) * subset1Y.size / dataSetY.size)

        if GiniGain > bestGiniGain:
            bestGiniGain = GiniGain
            bestFeature = i

    return bestFeature

def GetSplit_BST(dataSetX, dataSetY,weight):
    index = ChooseBestFeatureToSplit_BST(dataSetX, dataSetY,weight) 
    left,right = testsplit_BST(index, dataSetX, dataSetY,weight)
    return {'index':index, 'left':left, 'right': right}

def TerminalMajorityCount_BST(labels,weight):
    if sum(weight[labels==1])> sum(weight[labels==0]) :
        return 1
    else:
        return 0

def split_BST(node, max_depth, min_size, depth):
    left = node['left']
    right = node['right']

    if len(left[0]) == 0 and len(right[0]) != 0:
        node['left'] = TerminalMajorityCount_BST(right[1], right[2])
        node['right'] = TerminalMajorityCount_BST(right[1], right[2])
    if len(left[0]) != 0 and len(right[0]) == 0:
        node['left'] = TerminalMajorityCount_BST(left[1], left[2])
        node['right'] = TerminalMajorityCount_BST(left[1], left[2])
        return
    if depth >= max_depth:
        node['right'], node['left'] = TerminalMajorityCount_BST(right[1], right[2]), TerminalMajorityCount_BST(left[1],
                                                                                                               left[2])

        return
    if len(left[0]) <= 10:
        node['left'] = TerminalMajorityCount_BST(left[1], left[2])
        if len(right[0]) <= 10:
            node['right'] = TerminalMajorityCount_BST(right[1], right[2])
            return
        else:
            node['right'] = GetSplit_BST(right[0], right[1], right[2])

            split_BST(node['right'], max_depth, min_size, depth + 1)
    else:
        node['left'] = GetSplit_BST(left[0], left[1], left[2])

        split_BST(node['left'], max_depth, min_size, depth + 1)
        if len(right[0]) <= 10:
            node['right'] = TerminalMajorityCount_BST(right[1], right[2])
            return
        else:
            node['right'] = GetSplit_BST(right[0], right[1], right[2])

            split_BST(node['right'], max_depth, min_size, depth + 1)

def build_tree_BST(trainX,trainY,weight,max_depth, min_size):
    root = GetSplit_BST(trainX,trainY,weight)
    split_BST(root, max_depth, min_size, 1)
    return root

def PredictionOfOne_BST(node, x):
    if x[node['index']] == 1:
        if isinstance(node['right'], dict):
            return PredictionOfOne_BST(node['right'], x)
        else:
            return node['right']
    else:
        if isinstance(node['left'], dict):
            return PredictionOfOne_BST(node['left'], x)
        else:
            return node['left']

def TestDT_BST(trainX, trainY, weight, testX, testY):
    mytree = build_tree_BST(trainX, trainY, weight, 10, 10)
    predictions = []
    count = 0
    for row in testX:
        prediction = PredictionOfOne_BST(mytree, row)
        predictions.append(prediction)
    for i in range(len(predictions)):
        if predictions[i] != testY[i]:
            count += 1
    loss = count / len(testY)
    return loss

def UpdateWeight(trainX, trainY, OLD_weight, testX, testY, alpha):
    mytree = build_tree_BST(trainX, trainY, OLD_weight, 10, 10)
    predictions = []
    newweight = np.zeros(len(OLD_weight))
    for row in testX:
        prediction = PredictionOfOne_BST(mytree, row)
        predictions.append(prediction)
    for i in range(len(predictions)):
        if predictions[i] != testY[i]:
            newweight[i] = OLD_weight[i] * np.e ** alpha
        else:
            newweight[i] = OLD_weight[i] * np.e ** (-alpha)
    NEW_weight = newweight / sum(newweight)
    return NEW_weight

def DT_BST(trainX, trainY,testX,testY,weight,max_depth, min_size):
    alpha = 0
    Alpha = []
    trees = []
    for i in range(50):
        tree = build_tree_BST(trainX, trainY,weight,10,10)
        trees.append(tree)
        ##calculate alpha:
        predictions = []
        newweight = np.zeros(len(weight))
        for row in testX:
            prediction = PredictionOfOne_BST(tree, row)
            predictions.append(prediction)
        wrongweight = []
        for i in range(len(predictions)):
            if predictions[i] !=  testY[i]:
                wrongweight.append(weight[i])
        errorrate = sum(wrongweight)
        alpha = math.log((1-errorrate)/errorrate)
        Alpha.append(alpha)
        ###update weight:
        for i in range(len(predictions)):
            if predictions[i] !=  testY[i]:
                newweight[i] = weight[i] * np.e**alpha
            else:
                newweight[i] = weight[i] * np.e**(-alpha)
        weight= newweight/sum(newweight)
    return trees,Alpha

def Test_DT_BST(trainX, trainY, testX, testY, weight, max_depth, min_size):
    BSTDS_Tree = DT_BST(trainX, trainY, testX, testY, weight, max_depth, min_size)
    pred_all = []
    for eachtree in BSTDS_Tree[0]:
        pred_all_for_one_tree = []
        for row in testX:
            pred_of_one_tree = PredictionOfOne(eachtree, row)
            pred_all_for_one_tree.append(pred_of_one_tree)
        pred_all.append(pred_all_for_one_tree)
    pred_all_array = np.asarray(pred_all).T
    result = []
    alphaarray_normalized = np.asarray(BSTDS_Tree[1]) / sum(BSTDS_Tree[1])
    multiply = pred_all_array * alphaarray_normalized
    for i in range(len(pred_all_array)):
        if sum(multiply[i]) > 0.5:
            result.append(1)
        else:
            result.append(0)
    count = 0
    for i in range(len(result)):
        if result[i] != testY[i]:
            count += 1
    loss = count / len(testY)
    return loss

def main():
    if len(sys.argv) == 4:
        train_data_file = sys.argv[1]
        test_data_file = sys.argv[2]
        Algorithm = sys.argv[3]
        train_data = read_dataset(train_data_file)
        test_data = read_dataset(test_data_file)
        dataset_train = generate_vectors(train_data, get_most_commons(train_data))
        dataset_test = generate_vectors(test_data, get_most_commons(test_data))
        dataset_trainX = dataset_train[0][:2000]
        dataset_trainY = dataset_train[1][:2000]
        dataset_testX = dataset_test[0][:2000]
        dataset_testY = dataset_test[1][:2000]
        weight_Initial = np.asarray([1 / len(dataset_trainX)] * len(dataset_trainX))
        if int(Algorithm) == 1:
            zero_one_loss_DT = TestDT(dataset_trainX, dataset_trainY, dataset_testX, dataset_testY)
            print('ZERO-ONE-LOSS-DT ' + str(zero_one_loss_DT))
        elif int(Algorithm) == 2:
            zero_one_loss_BT = Test_BAG_DT(dataset_trainX, dataset_trainY, dataset_testX, dataset_testY)
            print('ZERO-ONE-LOSS-BT ' + str(zero_one_loss_BT))
        elif int(Algorithm) == 3:
            zero_one_loss_RF = Test_RF(dataset_trainX,dataset_trainY,dataset_testX,dataset_testY)
            print('ZERO-ONE-LOSS-RF ' + str(zero_one_loss_RF))
        elif int(Algorithm) == 4:
            zero_one_loss_BST = Test_DT_BST(dataset_trainX,dataset_trainY,dataset_testX,dataset_testY, weight_Initial, 10, 10)
            print('ZERO-ONE-LOSS-BST ' + str(zero_one_loss_BST))
    else:
        print('usage: python hw4.py train.csv test.csv')
        print('exiting...')

main()


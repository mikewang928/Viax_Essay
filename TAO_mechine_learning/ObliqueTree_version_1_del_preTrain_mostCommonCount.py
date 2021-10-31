#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 14:52:42 2021

@author: mikewang
"""

#%% import packages 
from collections import Counter
import numpy as np
from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split
#from keras.datasets import mnist

#%%dataset to test 
#generating the dumy testing dataset  
data = datasets.load_digits()
X, y = data.data, data.target

#(train_X, train_y), (test_X, test_y) = mnist.load_data()

#X_pre_train = np.random.randint()
#dummy_pretrain_label_internal = np.random.randint(np.min(X),np.max(X),size=(X.shape[0],X.shape[1])) # internal node labels
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=1234
)
data_dimension = X.shape[1]
label_position = data_dimension + 1

import tensorflow as tf
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train=X_train.reshape(X_train.shape[0], -1)
X_test=X_test.reshape(X_test.shape[0], -1)
from sklearn.linear_model import LogisticRegression,LinearRegression
import warnings
warnings.filterwarnings("ignore")

class LeafPredictor():
    def __init__(self, y=0):
        self.y = y
    def fit(self, X, y):
        counts = np.bincount(y)
        self.y = np.argmax(counts)
    def predict(self, X):
        return np.ones(X.shape[0])*self.y


#%% node class 
class Node:
    #value is the keyword only value 
    def __init__(self, left=None, right=None, parent = None, *,split_func=None, indicator=None): 
        self.left = left
        self.right = right
        self.parent = parent
        #self.indicator = indicator
        self.split_func = LeafPredictor()
        # self.split_func = svm.SVC(kernel='linear')
        self.predictor = 0 # when splitfunction can not be used

    def check_split(self):
        if not self.is_leaf_node():
            self.split_func = LogisticRegression(max_iter = 100)

    def fit(self, X, y):
        # if no label inputs go left
        if len(y) == 0:
            self.predictor = 0
        # if all label are the same, than predict that label
        elif len(np.unique(y))==1:
            self.predictor = y[0]
        # if normally trainned then train split_function
        else:
            self.split_func.fit(X, y)
            self.predictor=None

    def predict(self, X):
        # if no input return an empty array
        if X.shape[0]==0:
            return np.array([])
        # if predictor exist than predict as the predictor
        if self.predictor is not None:
            return np.ones(X.shape[0])*self.predictor
        # else predict as the split function predicts
        else:
            return self.split_func.predict(X)
 
    # check if it is a leaf node ture if it contains a value, false if does not contains a value
    def is_leaf_node(self):
        return self.left is None
    
#%% obli que tree
class tree:
    '''
    class initialization
    '''
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.root = self._init_tree()
    
    '''
    fit function
    '''    
    def fit(self, X_train, y_train, X_test, y_test, iteration):
        self._pre_train_tree(X_train, y_train, X_test, y_test)
        
        
        self._train_tree(X_train, y_train, X_test, y_test, iteration)
        
    #####################################################################################
    #                                  init tree method                                 # 
    #####################################################################################       
    
    #-------------------------helper method which init the tree-------------------------#
    # 1) we initalized the tree structure with certain depth full binary tree           #
    # 2) then we pretrain the tree to initalized the split_fun(i.e. from none to svm)   #
    #-----------------------------------------------------------------------------------#
    def _init_tree(self, depth=0):
        if depth >= self.max_depth:
            inited_tree = Node()
            inited_tree.check_split()
        else:
            left = self._init_tree(depth+1)
            right = self._init_tree(depth+1)
            inited_tree = Node(left, right)
            left.parent = inited_tree
            right.parent = inited_tree
            inited_tree.check_split()
        return inited_tree
 
    def _traverse(self):
        internals, leaves, parents = [], [], []
        nodes = [self.root]
        while len(nodes)>0:
            n = nodes.pop()# here when pop the poped item will not be in the list anymore
            if n is not None:
                nodes.append(n.left)
                nodes.append(n.right)
                if n.left is not None:
                    internals.append(n)
                else:
                    leaves.append(n)
                if n.parent is not None: 
                    if not n.parent in parents: 
                        parents.append(n.parent)
        return internals,leaves, parents

    #-------------------------helper method which init the split_func---------------------------------#
    # 1) we first generate a block of data which consist of the same format as the training data set  #
    # 2) then we pass them into the Nodes and train the split function i.s. SKlearn's SVM model       #
    # 3) when reaching the leaf node we count the most common label and determine leaf node label     #
    #-------------------------------------------------------------------------------------------------# 
    def _pre_train_tree(self, X, y, X_test, y_test):
        # pretrain dataset
        nodes, X_all, y_all = [self.root], [X], [y]
        while len(nodes)>0:
            n_temp, X_temp, y_temp = nodes.pop(), X_all.pop(), y_all.pop()
            # if internal node
            if n_temp.left is not None:
                if len(X_temp)>0 and len(np.unique(y_temp)) > 1:
                    # reconsruct the inital labels for the internal nodes
                    #print("================ internal node =================")
                    #print(n_temp)
                    #print("--- node parent")
                    #print(n_temp.parent)
                    #print('--- X_temp.shape ')
                    #print(X_temp.shape)
                    uni = np.unique(y_temp)
                    #print("--- uni_Y_temp.shape ")
                    #print(uni.shape)
                    #print(uni)
                    psudo = np.copy(y_temp)
                    for unis in uni[:int(len(uni) / 4)]:
                        psudo[psudo == unis] = 0
                    for unis in uni[int(len(uni) / 4):]: 
                        psudo[psudo == unis] = 1 
                    n_temp.fit(X_temp, psudo) # psudo is the internal node label
                    pred = n_temp.predict(X_temp)
                    '''
                    if len(X_temp[pred == 0]) == 0 or len(X_temp[pred == 1]) == 0 or len(np.unique(y_temp[pred == 0])) == 1 or len(np.unique(y_temp[pred == 1])) == 1:
                        #print("--- internal node to leaf node!")
                        n_temp.left = n_temp.right = None
                        n_temp.fit(X_temp, y_temp)
                    else:
                    '''
                    # inside the X_all, the first half is the dataset who have pred == 0
                    X_all.append(X_temp[pred == 0])
                    y_all.append(y_temp[pred == 0])
                    # the second half is the dataset who have pred == 1 
                    X_all.append(X_temp[pred == 1])
                    y_all.append(y_temp[pred == 1])
                    nodes.append(n_temp.left)
                    nodes.append(n_temp.right)
                    
            # if leaf node
            elif n_temp.left is None:
                #print("================ leaf node =================")
                #print(n_temp)
                #print("--- node parent")
                #print(n_temp.parent)
                #print('--- X_temp.shape ')
                #print(X_temp.shape)
                uni = np.unique(y_temp)
                #print("--- uni_Y_temp.shape ")
                #print(uni.shape)
                #print(uni)
                n_temp.fit(X_temp, y_temp)
                print("leaf node label", n_temp.split_func.y)
        train_pred = self._tree_predict(X_train, self.root)
        test_pred = self._tree_predict(X_test, self.root)
        train_acc = self._tree_accuracy(y_train, train_pred)
        test_acc = self._tree_accuracy(y_test, test_pred)
        print("++++++++++++++++++++ train_iter:0 train_acc:" + str(train_acc) + " test_acc:"+str(test_acc))
        
    #####################################################################################
    #                                  tree prediction                                  # 
    #####################################################################################
    # trancerse tree 
    def _tree_predict(self, X, node):
        return np.ravel(np.array([self._pred_traverse_tree(x, node) for x in X]))
    
    
    def _pred_traverse_tree(self, X, node):
        #print(X.shape)
        nodes = [node]
        #print(ind_all)
        while len(nodes) > 0:
            n_temp = nodes.pop()
            #print(X_temp.shape)
            #print(X.reshape(1, -1).shape)
            pred = n_temp.predict(X.reshape(1, -1))
            # when internal 
            if n_temp.left is not None:
                #X_all.append(X_temp[pred == 0])
                #X_all.append(X_temp[pred == 1])
                if pred == 0:
                    nodes.append(n_temp.left)
                elif pred == 1:
                    nodes.append(n_temp.right)
            else:
                prediction = pred
        return prediction

    #####################################################################################
    #                                  train tree method                                # 
    #####################################################################################        
    
    #-------------------------helper method which train the tree-------------------------#
    # 1) we first put all data into the left child of the tree and to see the result     #
    # 2) we then feed all data into the right child of the tree and see the prediction   #
    # 3) we then we then compare left and right                                          #
    #       -> if left is better than right, we assign the dataset a weight of 0         #
    #       -> if right is better than left, we assign the dataset a weight of 1         #  
    #       -> if left and right are equal, we assign that data a weight of 2            #
    # 4) we use the weights as labels to train the split functions in the internal node  #
    # 5) repeat above steps until we meet the leaf                                       #
    #------------------------------------------------------------------------------------#
    
    def _train_tree(self, X_train, y_train, X_test, y_test, iterations):
        inter = 0
        while inter < iterations:
            #internal_predictions = []
            nodes, X_all, y_all = [self.root], [X_train], [y_train]
            while len(nodes) > 0:
                n_temp, X_temp, y_temp = nodes.pop(), X_all.pop(), y_all.pop()
                
                # ==================================== if internal node ==================================
                if n_temp.left is not None:
                    internal_label_train_ind =[]
                    # put all data point to the left child 
                    left_internal_prediction =  self._tree_predict(X_temp, n_temp.left)
                    # put all data point to the right child 
                    right_internal_prediction = self._tree_predict(X_temp, n_temp.right)
                    # check if the left prediction equals to the label or not (equal = 1, not equal = 0)
                    left_internal_prediction_copy = np.copy(left_internal_prediction)
                    right_internal_prediction_copy = np.copy(right_internal_prediction)
                    left_internal_prediction_copy[left_internal_prediction == y_temp] = 1 
                    left_internal_prediction_copy[left_internal_prediction != y_temp] = 0
                    #print("-- left internal prediction correct: " + str(len(left_internal_prediction_copy[left_internal_prediction_copy == 1])))
                    #print("-- left internal prediction incorrect: " + str(len(left_internal_prediction_copy[left_internal_prediction_copy == 0])))
                    # check if the right prediction equals to the label or not (equal = 1, not equal = 0)
                    right_internal_prediction_copy[right_internal_prediction == y_temp] = 1
                    right_internal_prediction_copy[right_internal_prediction != y_temp] = 0
                    #print("-- right internal prediction correct: " + str(len(right_internal_prediction_copy[right_internal_prediction_copy == 1])))
                    #print("-- right internal prediction incorrect: " + str(len(right_internal_prediction_copy[right_internal_prediction_copy == 0])))
                    # assigning the internal train label (1 left, 0 don't care, -1 right)
                    internal_label_train = left_internal_prediction_copy.astype(np.int16) - right_internal_prediction_copy.astype(np.int16) # 1 left, 0 don't care, -1 right
                    #print(left_internal_prediction_copy[:100])
                    #print(right_internal_prediction_copy[:100])
                    #print(internal_label_train[:100])
                    #print(len(internal_label_train[internal_label_train==1]))
                    #print(len(internal_label_train[internal_label_train==-1]))
                    internal_label_train_copy = np.copy(internal_label_train)
                    internal_label_train_copy[internal_label_train == 0] = 2 # don't care
                    internal_label_train_copy[internal_label_train == 1] = 0 # left better
                    internal_label_train_copy[internal_label_train == -1] = 1 # right better 
                    
                    # extracting care data indexs
                    for i, internal_l_train in enumerate(internal_label_train_copy):
                        if internal_l_train == 0 or internal_l_train == 1:
                            internal_label_train_ind.append(i)
                    '''        
                    #print("=====================================================")
                    #print("X_temp.shape: (" + str(X_temp.shape[0]) +", "+ str(X_temp.shape[1])+")")
                    #print("internal_label_train_ind shape: " + str(len(internal_label_train_ind)))
                    '''
                    # -------------------------------  trainning internal node --------------------------------------
                    # fitting the model with cared data
                    #print("--- care points size: " + str(len(internal_label_train_ind)))
                    #print("--- care points label type: " + str(np.unique(internal_label_train_copy[internal_label_train_ind])))
                    #temp111 = n_temp.predict(X_temp)
                    #print("wrong samples:",sum(abs(temp111[internal_label_train_ind]-internal_label_train_copy[internal_label_train_ind])))

                    n_temp.fit(X_temp[internal_label_train_ind],internal_label_train_copy[internal_label_train_ind])
                    internal_label_pred = n_temp.predict(X_temp)
                    #print("after train prediction should go to left: " + str(len(internal_label_pred[internal_label_pred == 0])))
                    #print("after train prediction should go to right: " + str(len(internal_label_pred[internal_label_pred == 1])))
                    X_all.append(X_temp[internal_label_pred == 0])
                    X_all.append(X_temp[internal_label_pred == 1])
                    y_all.append(y_temp[internal_label_pred == 0])
                    y_all.append(y_temp[internal_label_pred == 1])
                    nodes.append(n_temp.left)
                    nodes.append(n_temp.right)
                # ==================================== if leaf node ==================================
                else:
                    n_temp.fit(X_temp, y_temp)
                    print("leaf node label", n_temp.split_func.y)
            train_pred = self._tree_predict(X_train, self.root)
            test_pred = self._tree_predict(X_test, self.root)
            train_acc = self._tree_accuracy(y_train, train_pred)
            test_acc = self._tree_accuracy(y_test, test_pred)
            inter += 1
            print("++++++++++++++++++++ train_iter:" + str(inter) + " train_acc:" + str(train_acc) + " test_acc:"+str(test_acc))
    
    
    
    def _tree_accuracy(self, y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy       

#%% tree tunnable hyper-parameter
max_train_iterations = 10
max_tree_depth = 3

#%% tree main

print("**************************** tree condition init ***************************")
clf = tree(max_depth = max_tree_depth)
i_init, l_init, p_init = clf._traverse()
print(len(i_init),len(l_init), len(p_init))

# starts trainning 
print("**************************** tree trainning starts ***************************")
clf.fit(X_train, y_train, X_test, y_test, max_train_iterations) 

print("**************************** tree condition after train ***************************")
i_post, l_post, p_post = clf._traverse()
print(len(i_post),len(l_post), len(p_post))

print("############ result comparsion #############")
#print(y_pred.reshape(y_pred.shape[0],))
#print(y_test)
y_pred = clf._tree_predict(X_test, clf.root)
print(y_pred.reshape(y_pred.shape[0],) == y_test)


#%% test field
'''
print("++++++++++++++++++++++++++++ test field cases +++++++++++++++++++++++++++")
print("**************************** tree condition init ***************************")
clf_test = tree(max_depth = max_tree_depth)
i_init, l_init, p_init = clf_test._traverse()
print(len(i_init),len(l_init), len(p_init))

# starts trainning 
print("######################### tree pre-train #########################")
clf_test._pre_train_tree(X_train, y_train) 
print("**************************** tree condition after pre-train ***************************")
i_pre_post, l_pre_post, p_pre_post = clf_test._traverse()
print(len(i_pre_post),len(l_pre_post), len(p_pre_post))

print("######################### tree train #########################")
clf_test._train_tree(X_train, y_train, X_test, y_test, max_train_iterations) 
print("**************************** tree condition after pre-train ***************************")
i_post, l_post, p_post = clf_test._traverse()
print(len(i_post),len(l_post), len(p_post))

#%% experiment 
left_prediction = np.array([1, 1, 0, 0, 1, 0, 1, 1])
right_prediction = np.array([1, 0, 1, 0, 1, 1, 0, 0])
split = left_prediction - right_prediction

print(left_prediction)
print(right_prediction)
print(split)

split[split == 0] = 2
split[split == 1] = 0
split[split == -1] = 1
print(split)
s_ind = []
for i, s in enumerate(split):
    if s == 0 or s == 1:
        s_ind.append(i)
print(s_ind)
print(split[s_ind])
'''
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

#%% node class 
class Node:
    #value is the keyword only value 
    def __init__(self, left=None, right=None, parent = None, *,split_func=None, indicator=None): 
        self.left = left
        self.right = right
        self.parent = parent
        #self.indicator = indicator
        self.split_func = split_func
 
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
        self._pre_train_tree(X_train, y_train)
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
        else:
            left = self._init_tree(depth+1)
            right = self._init_tree(depth+1)
            inited_tree = Node(left, right)
            left.parent = inited_tree
            right.parent = inited_tree
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
    def _pre_train_tree(self, X, y):
        # pretrain dataset
        nodes, X_all, y_all = [self.root], [X], [y]
        while len(nodes)>0:
            n_temp, X_temp, y_temp = nodes.pop(), X_all.pop(), y_all.pop()
            n_temp.split_func = svm.SVC(kernel='linear')
            # if internal node
            if n_temp.left is not None:
                if len(X_temp)>0 and len(np.unique(y_temp)) > 1:
                    # reconsruct the inital labels for the internal nodes
                    #print("================ node =================")
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
                    for unis in uni[:int(len(uni) / 2)]:
                        psudo[psudo == unis] = 0
                    for unis in uni[int(len(uni) / 2):]: 
                        psudo[psudo == unis] = 1
                    #print("--- Y_temp after internal label assignment ")
                    #print(np.unique(psudo))
                    #psudo[psudo == uni[:int(len(uni) / 2)]] = 0 # the first half of the leaf label are setted to 0 
                    #psudo[psudo == uni[int(len(uni) / 2):]] = 1 # the second half of the leaf label are setted to 1 
                    n_temp.split_func.fit(X_temp, psudo) # psudo is the internal node label
                    pred = n_temp.split_func.predict(X_temp)
                    if len(X_temp[pred == 0]) == 0 or len(X_temp[pred == 1]) == 0 or len(np.unique(y_temp[pred == 0])) == 1 or len(np.unique(y_temp[pred == 1])) == 1:
                        #print("--- internal node to leaf node!")
                        n_temp.left = n_temp.right = None
                        n_temp.split_func.fit(X_temp, y_temp)
                    else: 
                        # inside the X_all, the first half is the dataset who have pred == 0
                        X_all.append(X_temp[pred == 0])
                        y_all.append(y_temp[pred == 0])
                        # the second half is the dataset who have pred == 1 
                        X_all.append(X_temp[pred == 1])
                        y_all.append(y_temp[pred == 1])
                        nodes.append(n_temp.left)
                        nodes.append(n_temp.right)
                    
                    
            elif n_temp.left is None:
                #print("******************** leaf node *******************")
                #print(n_temp)
                #print("*** leaf node parent")
                #print(n_temp.parent)
                #print("*** number of X_temp: "  +str(len(X_temp)))
                #print("*** unique temp in leaf node")
                #print(np.unique(y_temp))
                n_temp.split_func.fit(X_temp, y_temp) 
                
    #####################################################################################
    #                                  tree prediction                                  # 
    #####################################################################################
    # trancerse tree 
    def _tree_predict(self, X, node):
        return np.array([self._pred_traverse_tree(x, node) for x in X])
    
    
    def _pred_traverse_tree(self, X, node):
        #print(X.shape)
        nodes = [node]
        #print(ind_all)
        while len(nodes) > 0:
            n_temp = nodes.pop()
            #print(X_temp.shape)
            #print(X.reshape(1, -1).shape)
            pred = n_temp.split_func.predict(X.reshape(1, -1))
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
    #       -> if left is better than right, we assign the dataset a weight of 1         #
    #       -> if right is better than left, we assign the dataset a weight of 2         #  
    #       -> if lef
    #          t and right are equal, we assign that data a weight of 0            #
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
                internal_label_train = np.copy(y_temp)
                
                # ==================================== if internal node ==================================
                if n_temp.left is not None:
                    # -------------------------------  setting internal train label --------------------------------------
                    internal_label_train_ind =[]
                    # put all data point to the left child 
                    left_internal_prediction =  self._tree_predict(X_temp, n_temp.left)
                    # put all data point to the right child 
                    right_internal_prediction = self._tree_predict(X_temp, n_temp.right)
                    # check if the left prediction equals to the label or not (equal = 1, not equal = 0)
                    left_internal_prediction[left_internal_prediction == y_temp.reshape(-1,1)] = 1 
                    left_internal_prediction[left_internal_prediction != y_temp.reshape(-1,1)] = 0
                    # check if the right prediction equals to the label or not (equal = 1, not equal = 0)
                    right_internal_prediction[right_internal_prediction == y_temp.reshape(-1,1)] = 1
                    right_internal_prediction[right_internal_prediction != y_temp.reshape(-1,1)] = 0
                    # assigning the internal train label (1 left, 0 don't care, -1 right)
                    internal_label_train = left_internal_prediction - right_internal_prediction # 1 left, 0 don't care, -1 right
                    internal_label_train[internal_label_train == 0] = 2 # don't care
                    internal_label_train[internal_label_train == 1] = 0 # left better
                    internal_label_train[internal_label_train == -1] = 1 # right better 
                    # extracting care data indexs
                    for i, internal_l_train in enumerate(internal_label_train):
                        if internal_l_train == 0 or internal_l_train == 1:
                            internal_label_train_ind.append(i)
                    '''        
                    #print("=====================================================")
                    #print("X_temp.shape: (" + str(X_temp.shape[0]) +", "+ str(X_temp.shape[1])+")")
                    #print("internal_label_train_ind shape: " + str(len(internal_label_train_ind)))
                    '''
                    # -------------------------------  trainning internal node --------------------------------------
                    # if all don't care or all perfered to one branch cut the tree
                    if len(internal_label_train_ind) == 0 or len(np.unique(internal_label_train[internal_label_train_ind])) == 1:
                        n_temp.left = n_temp.right = None
                        n_temp.split_func.fit(X_temp, y_temp)
                    else:    
                        #print("internal_label_train.unique: " + str(np.unique(internal_label_train[internal_label_train_ind])))
                        n_temp.split_func.fit(X_temp[internal_label_train_ind],np.ravel(internal_label_train[internal_label_train_ind]))
                        internal_label_pred = n_temp.split_func.predict(X_temp)
                        X_all.append(X_temp[internal_label_pred == 0])
                        X_all.append(X_temp[internal_label_pred == 1])
                        y_all.append(y_temp[internal_label_pred == 0])
                        y_all.append(y_temp[internal_label_pred == 1])
                        nodes.append(n_temp.left)
                        nodes.append(n_temp.right)
                # ==================================== if leaf node ==================================
                else:
                    n_temp.split_func.fit(X_temp, y_temp)
            train_pred = self._tree_predict(X_train, self.root)
            test_pred = self._tree_predict(X_test, self.root)
            train_acc = self._tree_accuracy(y_train, train_pred)
            test_acc = self._tree_accuracy(y_test, test_pred)
            inter += 1
            print("-------------train_iter:" + str(inter) + " train_acc:" + str(train_acc) + " test_acc:"+str(test_acc))
    
    
    
    def _tree_accuracy(self, y_true, y_pred):
        accuracy = np.sum(y_true.reshape(-1, 1) == y_pred) / len(y_true)
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

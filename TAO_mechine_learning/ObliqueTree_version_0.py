# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 18:08:52 2021

@author: wsycx
"""

#%% import packages 
from collections import Counter
import numpy as np
from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split


#%%dataset to test 
#generating the dumy testing dataset  
data = datasets.load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)
data_dimension = X.shape[1]
label_position = data_dimension + 1

#%% node class 
class Node:
    #value is the keyword only value 
    def __init__(self, left=None, right=None, *,split_func=None, indicator=None): 
        self.left = left
        self.right = right
        self.indicator = indicator
        self.split_func = split_func
 
    # check if it is a leaf node ture if it contains a value, false if does not contains a value
    def is_leaf_node(self):
        return self.indicator is not None
    
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
    def fit(self,X,y):
                
        self.root = self._train_tree(X,y)
        
    #####################################################################################
    #                                  init tree method                                 # 
    #####################################################################################       
    
    #-------------------------helper method which init the tree-------------------------#
    # 1) we initalized the tree structure with certain depth full binary tree           #
    # 2) then we pretrain the tree to initalized the split_fun(i.e. from none to svm)   #
    #-----------------------------------------------------------------------------------#
    def _init_tree(self, depth=0): 
        left = self._init_tree(depth+1)
        right = self._init_tree(depth+1)
        if depth >= self.max_depth: 
            inited_tree = Node(left, right)
            dummy_pretrain_train = np.random.randint(1,10,size=(10000,data_dimension))
            dummy_pretrain_label_leaf = np.random.randint(1,10,size=(10000,1))
            dummy_pretrain_data = np.concatenate((dummy_pretrain_train,dummy_pretrain_label_leaf),axis=1)
            tree = self._pre_train_tree(dummy_pretrain_data)
        return tree
    
    #-------------------------helper method which init the split_func---------------------------------#
    # 1) we first generate a block of data which consist of the same format as the training data set  #
    # 2) then we pass them into the Nodes and train the split function i.s. SKlearn's SVM model       #
    # 3) when reaching the leaf node we count the most common label and determine leaf node label     #
    #-------------------------------------------------------------------------------------------------# 
    def _pre_train_tree(self, dummy_pretrain_data, curr_depth = 0):
        # pretrain dataset
        
        while self.max_depth >= curr_depth: 
            dataset_left = []
            dataset_right = []
            if self.max_depth > curr_depth:
                dummy_pretrain_label_internal = np.random.randint(0,2,size=(len(dummy_pretrain_data),)) # internal node labels
                self.root.split_func = svm.SVC(kernel = 'linear',max_iter= 10)
                self.root.split_func.fit(dummy_pretrain_data[:,:data_dimension],dummy_pretrain_label_internal)
                prediction = self.root.split_func.predict(dummy_pretrain_data[:,:data_dimension])
            elif  self.max_depth == curr_depth:
                self.root.split_func = svm.SVC(kernel = 'linear',max_iter= 10)
                self.root.split_func.fit(dummy_pretrain_data[:,:data_dimension],dummy_pretrain_data[:,label_position])
                self.root.indicator = 1
                prediction = self.root.split_func.predict(dummy_pretrain_data[:,:data_dimension])
            for i, items in enumerate(prediction):
                if items == 0: 
                    dataset_left.append(dummy_pretrain_data[i,:], axis=0)
                if items == 1: 
                    dataset_right.append(dummy_pretrain_data[i,:], axis=0)
            self.root.left._pre_train_tree(dataset_left, curr_depth + 1)
            self.root.left._pre_train_tree(dataset_right, curr_depth + 1)    
 
    #####################################################################################
    #                                  tree prediction                                  # 
    #####################################################################################
    # trancerse tree 
    def _tree_predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node() == False:
            predict_internal = node.split_func.predict(x)
            if predict_internal == 0:
                node.left._traverse_tree(x)
            if predict_internal == 1:
                node.right._traverse_tree(x)
        if node.is_leaf_node() == True: 
            predict_leaf = node.split_func(x)
        return predict_leaf

    #####################################################################################
    #                                  train tree method                                # 
    #####################################################################################        
    
    #-------------------------helper method which train the tree-------------------------#
    # 1) we first put all data into the left child of the tree and to see the result     #
    # 2) we then feed all data into the right child of the tree and see the prediction   #
    # 3) we then we then compare left and right                                          #
    #       -> if left is better than right, we assign the dataset a weight of 1         #
    #       -> if right is better than left, we assign the dataset a weight of 2         #  
    #       -> if left and right are equal, we assign that data a weight of 0            #
    # 4) we use the weights as labels to train the split functions in the internal node  #
    # 5) repeat above steps until we meet the leaf                                       #
    #------------------------------------------------------------------------------------#
    def _train_tree(self, X, y, current_depth = 0):
        dataset_with_label = np.concatenate((X,y),axis=1)
        while self.max_depth >= current_depth:
            left_right_train = []
            left_prediction_train = []
            right_prediction_train = []
            left_condition = []
            right_condition = []
            left_right_difference = []
            train_ind = []
            train_internal = []
            internal_weight =[]
            internal_label = []
            post_prediction = []
            data_left = []
            data_right = []
            if self.max_depth > current_depth:
                left_right_train = dataset_with_label
                # get perictions from the tree when putting all dataset on the left child
                left_prediction_train = self.root.left._tree_predict(left_right_train)
                # get perictions from the tree when putting all dataset on the right child
                right_prediction_train = self.root.right._tree_predict(left_right_train)
                # check if left prediction predicts the correct label
                for ind_left, left_pred_train in enumerate(left_prediction_train):
                    if left_pred_train == left_right_train(ind_left,label_position):
                        check = 1
                    else: 
                        check = 0
                    left_condition.append(check)
                # check if right prediction predicts the correct label    
                for ind_right, right_pred_train in enumerate(right_prediction_train):
                    if right_pred_train == left_right_train(ind_left,label_position):
                        check = 1
                    else: 
                        check = 0
                    right_condition.append(check)
                # find out the weight seperator
                left_right_difference = left_condition - right_condition # 1 left, 0 don't care, -1 right
                # distributes trainning index
                for ind_diff, diff in enumerate(left_right_difference):
                    if diff == 1:
                        mark = 0
                        train_ind.append(ind_diff)
                    elif diff == -1:
                        mark = 1
                        train_ind.append(ind_diff)
                    elif diff == 0:
                        mark = 3
                    internal_weight.append(mark)
                # constructing trainning matrixs 
                for ids in train_ind: 
                    train_internal.append(left_right_train[ids,:data_dimension], axis = 0)
                    internal_label.append(internal_weight[ids])
                # starts trainning
                self.root.split_func.fit(train_internal, internal_label)
                post_prediction = self.root.split_func.predict(left_right_train[:,:data_dimension])
            
            elif self.max_depth == current_depth:
                self.root.split_func.fit(left_right_train[:,:data_dimension],left_right_train[:,label_position])
                post_prediction = self.root.split_func.predict(left_right_train[:,:data_dimension])
                
                
            for post_data_ind, post_pred in enumerate(post_prediction):
                if post_pred == 1:
                    data_left.append(left_right_train[post_data_ind,:])
                if post_pred == 2:
                    data_right.append(left_right_train[post_data_ind,:])
            self.root.left._train_tree(data_left[:,:data_dimension], data_left[:,label_position], current_depth + 1)
            self.root.right._train_tree(data_right[:,:data_dimension], data_right[:,label_position], current_depth + 1)
                   
#%% Trainning test 
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


clf = tree(max_depth=10)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy(y_test, y_pred)

print("Accuracy:", acc)
        


        


     
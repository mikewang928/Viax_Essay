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
data = datasets.load_digits()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
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
    def fit(self,X,y):
        self._pre_train_tree(X, y)
        #self._train_tree(X,y)
        
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
                    print("================ node =================")
                    print(n_temp)
                    print("--- node parent")
                    print(n_temp.parent)
                    print('--- X_temp.shape ')
                    print(X_temp.shape)
                    uni = np.unique(y_temp)
                    print("--- uni_Y_temp.shape ")
                    print(uni.shape)
                    print(uni)
                    psudo = np.copy(y_temp)
                    for unis in uni[:int(len(uni) / 2)]:
                        psudo[psudo == unis] = 0
                    for unis in uni[int(len(uni) / 2):]: 
                        psudo[psudo == unis] = 1
                    print("--- Y_temp after internal label assignment ")
                    print(np.unique(psudo))
                    #psudo[psudo == uni[:int(len(uni) / 2)]] = 0 # the first half of the leaf label are setted to 0 
                    #psudo[psudo == uni[int(len(uni) / 2):]] = 1 # the second half of the leaf label are setted to 1 
                    n_temp.split_func.fit(X_temp, psudo) # psudo is the internal node label
                    pred = n_temp.split_func.predict(X_temp)
                    # inside the X_all, the first half is the dataset who have pred == 0
                    X_all.append(X_temp[pred == 0])
                    y_all.append(y_temp[pred == 0])
                    # the second half is the dataset who have pred == 1 
                    X_all.append(X_temp[pred == 1])
                    y_all.append(y_temp[pred == 1])
                    nodes.append(n_temp.left)
                    nodes.append(n_temp.right)
                if len(X_temp) == 0 or len(np.unique(y_temp)) == 1:
                    print("---### empty internal Node!")
                    n_temp.parent.left = n_temp.parent.right = None 
            elif n_temp.left is None:
                print("******************** leaf node *******************")
                print(n_temp)
                print("*** leaf node parent")
                print(n_temp.parent)
                print("*** number of X_temp : "  +str(len(X_temp)))
                if len(X_temp) > 0 and len(np.unique(y_temp)) > 1:
                    print("*** unique temp in leaf node")
                    print(np.unique(y_temp))
                    n_temp.split_func.fit(X_temp, y_temp)
                if len(X_temp) == 0 or len(np.unique(y_temp)) == 1: 
                    print("***### empty leaf Node!")
                    print(n_temp.parent)
                    n_temp.parent.left = n_temp.parent.right = None
        '''
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
        '''
    #####################################################################################
    #                                  tree prediction                                  # 
    #####################################################################################
    def predict(self, X):
        N = X.shape[0]
        predictions = np.zeros(N)
        nodes, ind_all = [self.root], [np.array([True]*N)]
        while len(nodes) > 0:
            n_temp, ind = nodes.pop(), ind_all.pop()
            pred = n_temp.split_func.predict(X[ind])
            if n_temp.left is not None:
                ind_all.append(pred == 0)
                ind_all.append(pred == 1)
                nodes.append(n_temp.left)
                nodes.append(n_temp.right)
            else:
                predictions[ind] = pred
        return predictions

    # trancerse tree 
    def _tree_predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, X):

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
    #       -> if lef
    #          t and right are equal, we assign that data a weight of 0            #
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

# testing if the tree has been buiit
clf = tree(max_depth=3)
i, l, p= clf._traverse()
print(len(i),len(l), len(p))

clf.fit(X_train, y_train)

#y_pred = clf.predict(X_test)
#acc = accuracy(y_test, y_pred)
#print("Accuracy:", acc)

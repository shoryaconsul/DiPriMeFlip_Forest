import numpy as np
import numpy.random as rn
from scipy.stats import mode
import random
from copy import deepcopy

## This file implements the DP-RF algorithm (Singh & Patil, 2014).
# The nodes store class fractions.

#%%################################################################
# Querying the value of a scalar attribute
## x: True atrribute value
## s: Sensitivity of attribute
## e: Privacy budget
def query(x,s,e):
    if s/e < 0:
        print('s = ',s,' e = ',e)
    return rn.laplace(x,s/e)

# Computing gini index for splitting on given attribute
## X: Data (M,N)
## y: Target (M,)
## a: Attrbiute (index)
## A: Categories for given attribute (List)
## C: List of class values (in target variable)
def compute_gini(X,y,a,A,C):
    N = len(y) # Number of data points
    Xa = X[:,a] # Selecting attribute from data
    
    gini = 0 # Accumulator for gini
    for j in A:
        ind_j = np.nonzero(Xa==j)[0] # Indices of data points in child j
        Nj = len(ind_j)
        if Nj > 0: # Non-empty child
            frac_j = np.array([np.count_nonzero(y[ind_j]==c)/Nj for c in C])
            gini = gini + Nj/N*(np.sum(frac_j**2)) # Add weighted, negative gini index
    return gini-1
#%% Tree class for DP-RF
class DPRT():
    def __init__(self, depth = 0, max_depth = np.inf, max_features = None, 
                 parent=None):
        
        self.frac = []
        self.count = None # For book-keeping
        
        self.split_ind = None   # Index of feature for split
        self.split_cat = None   # Categories for split
        self.max_features = max_features # Features to consider per split
        
        self.children = []
        self.parent = parent
        
        if depth > max_depth:
            raise ValueError('Depth larger than max depth')
        
        self.depth = depth      # Depth = 0 as tree is empty
        self.max_depth = max_depth
        
    #%% Fitting tree to data given target values - all nodes noised    
    ## A: Dictionary of values (for categorical) features. 
        # A only contains values for features that can be split on
    ## C: List of class values
    ## eps: Privacy budget for tree
    ## b_split: Fraction of privacy budget for determining split
    def fit(self,X,y,A,C,eps=None,b_split=0.5):
        # X: considered to have M samples and N features (M x N)
        # y: value to be predicted (M,)
        
        if eps is None:
            raise ValueError('Privacy parameters required')
        
        M,N = np.shape(X)
        self.count = int(M)
        
        eps_split = eps*b_split/(self.max_depth+1) # Privacy budget for split
        eps_count = eps*(1-b_split)/(self.max_depth+1) # Privacy budget for counts
    
        for c in C: # Finding class fractions
            N_c = query(np.count_nonzero(y==c),1,eps_count)
            N_c = max(0,N_c)
            self.frac.append(N_c)
        if sum(self.frac) == 0: # Empty node
            self.frac = [0,1]
        else:
            self.frac = [N_c/sum(self.frac) for N_c in self.frac]
                    
        
        if self.depth == self.max_depth or not A or (1 in self.frac):  # Leaf node
            return
        
        if self.max_features is None:
            K = int(N)
        else:
            K = int(self.max_features)
        
        K = min(K,len(A)) # In case there are less than K attributes
        idx_cand = rn.choice(list(A.keys()),K,replace=False) # Select subset of attributes
        
        gini = np.zeros(K)
        for i,a in enumerate(idx_cand):
            
            gini[i] = compute_gini(X,y,a,A[a],C)
        
        score = np.exp(0.5*gini*eps_split/1.)
        self.split_ind = rn.choice(idx_cand,p=score/np.sum(score))
        
        A_child = deepcopy(A)
        self.split_cat = A_child.pop(self.split_ind) # Can no longer split on chosen attribute
        
        # Splitting data
        for cat in self.split_cat:
            ind_cat = np.where(X[:,self.split_ind]==cat)[0] 
            tree_cat = DPRT(depth=self.depth+1,max_depth=self.max_depth,
                            max_features = self.max_features,parent=self)
            tree_cat.fit(X[ind_cat,:],y[ind_cat],A_child,C,eps,b_split)
            self.children.append(tree_cat)
            
        
    #%% Predicting target values based on attributes    
    ## X: M samples and N features (M x N)
    ## C: Categories of target variable (List)
    ## Returns predicted y
    def predict(self,X,C):        
        # if self.depth == 0 and self.split_ind is None:
        #     raise ValueError('Tree not fit to data')
        
        M,N = np.shape(X)
        y = np.zeros(M) # Predicted values
        
        for i in range(M):
            y[i] = self.predict_y(X[i,:],C)
        return y        
        
    # x: Sample of N features (N,)    
    def predict_y(self,x,C):
        
        ind = self.split_ind        
        if self.split_ind is None: # Leaf node
            return C[self.frac.index(max(self.frac))]
               
        # Go to appropriate child
        child_ind = self.split_cat.index(x[ind])
        return self.children[child_ind].predict_y(x,C)
    
#%%% DP-RF
# n_trees: Number of random trees
class DPRF():
    def __init__(self, n_trees = 10, max_depth = np.inf, max_features = None):
        self.num_trees = n_trees # Numm=ber of trees
        self.trees = [] # ExtRandTrees
                         
        for i in range(n_trees): # Initialize all trees
            self.trees.append(DPRT(depth=0,max_depth=max_depth,
                                   max_features=max_features,parent=None))
    
    #%% Fitting random tree to data given target values 
    # bstrap: Fraction of data to be drawn as a bootstrap sample
    def fit(self,X,y,A,C,bstrap=1,eps=None,b_med=0.5):
        
        M,N = np.shape(X)
        num_bstrap = int(np.floor(bstrap*M))
        idx_bstrap = rn.choice(M,num_bstrap,replace=False)
        
        for tree in self.trees:
            # print('Fitting tree')
            tree.fit(X[idx_bstrap,:],y[idx_bstrap],A,C,eps/self.num_trees,b_med)
            # print(tree.split_ind)

    #%% Predicting target values based on attributes    
    ## X: M samples and N features (M x N)
    ## C: List of class values
    ## Returns predicted y  
    def predict(self,X,C):
        M,N = np.shape(X)
        pred = np.zeros((M,self.num_trees))
        
        for i in range(self.num_trees): # Prediction from each tree
            pred[:,i] = self.trees[i].predict(X,C)
        
        return mode(pred,axis=1)[0][:,0] # Majority vote


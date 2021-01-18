import numpy as np
import numpy.random as rn
import random
from copy import deepcopy

# This file implements the proposed algorithm for regression (stores means).


# %%################################################################
# Querying the value of a scalar attribute
# x: True atrribute value
# s: Sensitivity of attribute
# e: Privacy budget
def query(x, s, e):
    if s/e < 0:
        print('s = ', s, ' e = ', e)
    return rn.laplace(x, s/e)


# Query median of continuous attribute
# Xj: Values of feature for each data point (M,)
# Xrange: Min and max value of Xj (2,)
# num: Number of points to choose from (Deprecated)
# eps: Privacy budget. None indicates no privatization
# Returns: Value of attribute to split at
def query_cont_med(Xj, Xrange, eps=None):
    M = np.shape(Xj)[0]
    # print('M: ',M)

    if M <= 1:
        return random.uniform(Xrange[0], Xrange[1])

    elif eps is None:  # Return true median
        return np.median(Xj)
    else:  # Private median
        score = np.abs(np.arange(-M, M+1, 2))  # Scoring function is N_L-N_R
        # print('SCORE: ', score)
        prob = np.exp(-0.5*eps*score)  # Probability disribution for median selection
        prob = prob/sum(prob)  # Normalizing distribution
        ind_med = rn.choice(M+1, p=prob)

        Xj_sort = np.sort(Xj)
        if ind_med == 0:
            x_pick = random.uniform(Xrange[0], Xj_sort[0])
        elif ind_med == M:
            x_pick = random.uniform(Xj_sort[-1], Xrange[1])
        else:
            x_pick = random.uniform(Xj_sort[ind_med-1], Xj_sort[ind_med])

        # h,x = np.histogram(Xj,bins=num+1,range=(Xrange[0],Xrange[1]))
        # N_left = np.cumsum(h)[:-1]

        # score = -np.abs(2*N_left - M)

        # prob = np.exp(0.5*eps*score)+1e-6 # Probability disribution for median selection
        # prob = prob/sum(prob) # Normalizing distribution
        # x_pick = rn.choice(x[1:-1],p=prob)

        return x_pick


# Query median of categorical attribute
# Xj: Values of feature for each data point (M,)
# Xdict: Possible values of Xj (list of variable length)
# eps: Privacy budget. None indicates no privatization
# Returns: List of attributes split for both children
def query_cat_med(Xj, Xdict, eps=None):
    M = np.shape(Xj)[0]
    Xnum = len(Xdict)

    unique, counts = np.unique(Xj, return_counts=True)
    num_val = np.zeros(Xnum)  # Number of data points with with each possible value
    i = 0
    for val in Xdict:
        if val in unique:
            num_val[i] = counts[unique == val][0]
        i = i+1

    # Function to convert decimal to logical lists
    blist = lambda i,n: [False]*(n-int(np.ceil(np.log2(i+1)))-int(i==0)
                                 ) + [bool(int(j)) for j in bin(i)[2:]]

    N_child = np.zeros(2**(Xnum-1)-1)  # Number of points in child node
    for i in range(1, 2**(Xnum-1)):  # Look at all possible split
        N_child[i-1] = np.sum(num_val[blist(i, Xnum)])

    if eps is None:  # Return true median
        i_min = np.argmin(np.abs(N_child-M/2))
        split_log = blist(i_min+1, Xnum)  # Membership to left or right chiled
        val_min = list(np.array(Xdict)[split_log])
        val_comp = list(set(Xdict)-set(val_min))

        return val_min, val_comp

    else:  # Private median
        score = -np.abs(2*N_child - M)
        prob = np.exp(0.5*eps*score)+1e-6  # Probability disribution for median selection
        prob = prob/sum(prob)  # Normalizing distribution
        i_pick = rn.choice(2**(Xnum-1)-1, p=prob)

        split_log = blist(i_pick+1, Xnum)  # Membership to left or right chiled
        val_pick = list(np.array(Xdict)[split_log])
        val_comp = list(set(Xdict)-set(val_pick))

        return val_pick, val_comp


# Function to find best split for continuous attributes
# X: Data (M,N)
# y: Target (M,)
# By: Max absolute value of target (1,)
# A: Dictionary of feature ranges (for cont)/values (for categorical)
# cat_idx: List of indices for categorical features
# K: Number of features to consider for split
# num_cand: Number of candidates for continuous split (Deprecated)
# eps: Privacy budget
# Returns: Splitting feature index, splitting value/categories
def split_count(X, y, A, By, cat_idx, K, eps=None):
    M, N = np.shape(X)
    K1 = min(K, len(list(A.keys())))  # In case there are less than K attributes
    idx_cand = rn.permutation(list(A.keys()))[:K1]  # Select k feature indices

    # Finding median splits
    val_cand = dict()
    if eps is None:
        eps_fn = None
    else:
        eps_fn = eps/2

    for idx in idx_cand:
        if idx in cat_idx:
            val_cand[idx] = query_cat_med(X[:, idx], A[idx], eps_fn)
        else:
            val_cand[idx] = query_cont_med(X[:, idx], A[idx], eps_fn)

    # Finding indices in children
    sse_idx = np.zeros(K1)
    for idx in idx_cand:

        if idx in cat_idx:
            ind_upp = np.where(X[:, idx] == np.expand_dims(
                np.array(val_cand[idx][0]), axis=1))[1]
            ind_low = np.where(X[:, idx] == np.expand_dims(
                np.array(val_cand[idx][0]), axis=1))[1]

        else:
            ind_upp = np.where(X[:, idx] >= val_cand[idx])[0]
            ind_low = np.where(X[:, idx] < val_cand[idx])[0]

        y_upp = y[ind_upp]
        y_low = y[ind_low]

        pos = np.where(idx_cand == idx)[0][0]
        if len(y_upp) != 0 and len(y_low) != 0:  # Checking that children are not empty
            sse_idx[pos] = (len(y_upp)*np.var(y_upp) + len(y_low)*np.var(y_low))/len(y)
        elif len(y_upp) == 0 and len(y_low) == 0:
            sse_idx[pos] = 40*By**2
        else:
            sse_idx[pos] = np.var(y)

    if eps is None:
        idx_split = idx_cand[np.argmin(sse_idx)]
    else:  # Exponential mechanism to pick split
        if len(y) != 0:
            score_split = np.exp(-0.5*sse_idx*eps_fn/(4*By**2/len(y)))
        else:  # Empty node, so all splits are equivalent
            score_split = np.ones(K1)
        # print('# SPLIT: ',len(y_upp),len(y_low))
        # if np.any(np.isnan(score_split/np.sum(score_split))):
        #    print('SSE ',sse_idx,' SCORE ',score_split)
        idx_split = rn.choice(idx_cand, p=score_split/np.sum(score_split))
    return idx_split, val_cand[idx_split]


# %% Base classes for Extremely Random Trees with splits along median

class DiPrimeTree():
    def __init__(self, depth=0, max_depth=np.inf, max_features=None,
                 parent=None):
        self.mean = np.nan
        self.count = 0

        self.split_ind = None   # Index of feature for split
        self.split_val = None   # Feature value at split
        self.max_features = max_features  # Features to consider per split

        self.left = None
        self.right = None
        self.parent = parent

        if depth > max_depth:
            raise ValueError('Depth larger than max depth')

        self.depth = depth      # Depth = 0 as tree is empty
        self.max_depth = max_depth

    # %% Fitting random tree to data given target values - only leaf nodes noised    
    # Left is larger than split, right is smaller than split
    # A: Dictionary of feature ranges (for cont)/values (for categorical)
        # features. A only contains values for features that can be split on
    # cat_idx: List of indices for categorical features
    # tbound: Lower and upper bound on target value or just maximum absolute value of bound
    # eps: Privacy budget for tree
    # b_med: Fraction of privacy budget for determining median split
    def fit(self, X, y, A, cat_idx, tbound, eps=None, b_med=0.5):
        # X: considered to have M samples and N features (M x N)
        # y: value to be predicted (M,)

        if eps is not None and b_med is None:
            raise ValueError('Budget split for median required')

        if tbound is not None:  # B_L, B_U are lower and upper bounds on y
            if len(tbound) == 2:
                B = np.abs(tbound[1]-tbound[0])  # Range of target value
                # mean_def = np.mean(tbound) # Default value of mean for empty node
                B_L = tbound[0]
                B_U = tbound[1]
            elif isinstance(tbound, int) or isinstance(tbound, float):
                B = 2*np.abs(tbound)
                # mean_def = 0 # Default value of mean for empty node
                B_L = -np.abs(tbound)
                B_U = np.abs(tbound)
            else:
                raise ValueError("Invalid value passed to tbound.")

            mean_def = random.uniform(B_L, B_U)

        M, N = np.shape(X)
        self.count = int(M)  # Storing count

        if eps is not None:
            eps_med = eps*b_med/self.max_depth  # Privacy budget for split
            eps_level = eps*(1-b_med)  # Privacy budget for mean
        else:
            eps_med = None

        if self.depth == self.max_depth or not A:  # Reached leaf node or no more attributes to split on
            # Private case
            if eps is not None:  # Compute and noise sufficient statistics 
                count_clip = len(y)
                #print('Depth: ',self.depth,' True count: ',len(y),' Stored count: ',self.count)

                if np.isnan(np.mean(y)):  # Empty node
                    self.mean = mean_def
                else:
                    query_mean = query(x=np.mean(y), s=B/count_clip, e=eps_level)
                    self.mean = max(B_L, min(query_mean, B_U))  # Bounding output in range of y
            # Non-private case
            else:  # Compute sufficient statistics
                if np.isnan(np.mean(y)):  # Empty node
                    self.mean = mean_def
                else:
                    self.mean = np.mean(y)
            return

        # Finding split
        # Number of features to consider at each split
        if self.max_features is None:
            K = int(N)
        else:
            K = int(self.max_features)

        By = np.amax(np.abs(tbound))
        # num_cand = 30 # Number of candidates

        self.split_ind, feat_val = split_count(X, y, A, By, cat_idx, K, eps_med)

        if self.split_ind in cat_idx:  # If categorical
            feat_left, feat_right = feat_val
            self.split_val = feat_left.copy()
        else:   # If continuous
            self.split_val = 1*feat_val

        # Splitting data
        if self.split_ind in cat_idx:
            ind_upp = np.where(X[:, self.split_ind] == np.expand_dims(
                np.array(feat_left), axis=1))[1]
            ind_low = np.where(X[:, self.split_ind] == np.expand_dims(
                np.array(feat_right), axis=1))[1]

            A_upp = deepcopy(A)
            if len(feat_left) == 1:
                A_upp.pop(self.split_ind, None)  # Remove from allowed splits
            else:
                A_upp[self.split_ind] = feat_left.copy()  # Update feature values

            A_low = deepcopy(A)
            if len(feat_right) == 1:
                A_low.pop(self.split_ind, None)  # Remove from allowed splits
            else:
                A_low[self.split_ind] = feat_right.copy()  # Update feature values

        else:
            ind_upp = np.where(X[:, self.split_ind] >= self.split_val)[0]
            ind_low = np.where(X[:, self.split_ind] < self.split_val)[0]

            A_upp = deepcopy(A)
            if A_upp[self.split_ind][1] <= self.split_val:  # If split no longer possible
                #print('POP: ',A_upp[self.split_ind][1],self.split_val)
                A_upp.pop(self.split_ind, None)
            else:
                A_upp[self.split_ind][0] = self.split_val  # Updating lower bound

            A_low = deepcopy(A)
            if A_low[self.split_ind][0] >= self.split_val:  # If split no longer possible
                #print('POP: ',A_low[self.split_ind][0],self.split_val)
                A_low.pop(self.split_ind, None)
            else:
                A_low[self.split_ind][1] = self.split_val  # Updating upper bound

        X_upp = X[ind_upp, :]
        y_upp = y[ind_upp]

        X_low = X[ind_low, :]            
        y_low = y[ind_low]

        # print('LEFT: ',len(y_upp),'RIGHT: ',len(y_low))

        # Recursively splitting
        tree_upp = DiPrimeTree(depth=self.depth+1, max_depth=self.max_depth,
                             max_features=self.max_features, parent=self)
        tree_upp.fit(X_upp, y_upp, A_upp, cat_idx, tbound, eps, b_med)
        self.left = tree_upp

        tree_low = DiPrimeTree(depth=self.depth+1, max_depth=self.max_depth,
                             max_features=self.max_features, parent=self)
        tree_low.fit(X_low, y_low, A_low, cat_idx, tbound, eps, b_med)

        self.right = tree_low

    # %% Predicting target values based on attributes
    # X: M samples and N features (M x N)
    # cat_idx: List of indices for categorical features
    # Returns predicted y
    def predict(self, X, cat_idx):
        if self.depth == 0 and self.split_ind is None:
            raise ValueError('Tree not fit to data')
        else:
            M, N = np.shape(X)
            y = np.zeros(M)  # Predicted values

            for i in range(M):
                y[i] = self.predict_y(X[i, :], cat_idx)
            return y

    # x: Sample of N features (N,)
    def predict_y(self, x, cat_idx):

        ind = self.split_ind

        if self.split_ind is None:  # Leaf node
            return self.mean

        # If true, go right
        if ind in cat_idx:
            dirn = (x[ind] not in self.split_val)
        else:
            dirn = x[ind] < self.split_val

        if not dirn:  # Go to right child
            return self.left.predict_y(x, cat_idx)
        else:  # Go to left child
            return self.right.predict_y(x, cat_idx)


# %############################################################################
# %% Extremely Random Forest
# n_trees: Number of random trees
# partition: If true, grow tree on disjoint subsets (rows) of data
class DiPrimeForest():
    def __init__(self, n_trees=10, max_depth=np.inf, max_features=None,
                 partition=True):
        self.num_trees = n_trees  # Numm=ber of trees
        self.trees = []  # ExtRandTrees
        self.partition = partition

        for i in range(n_trees):  # Initialize all trees
            self.trees.append(DiPrimeTree(depth=0, max_depth=max_depth,
                                          max_features=max_features,
                                          parent=None))

    # %% Fitting random tree to data given target values
    # Left is larger than split, right is smaller than split
    def fit(self, X, y, A, cat_idx, tbound, eps=None, b_med=0.5):

        if self.partition:  # Growing trees on disjoint subsets of rows
            M, N = np.shape(X)
            ind_part = np.array_split(rn.permutation(M), self.num_trees)  # Partition

            for i, tree in enumerate(self.trees):  # Fit tree to partition
                if eps is None:
                    tree.fit(X[ind_part[i], :], y[ind_part[i]], A, cat_idx, tbound)
                else:
                    tree.fit(X[ind_part[i], :], y[ind_part[i]], A, cat_idx, tbound, eps, b_med) 

        else:  # Growing trees on all the data
            for (t, tree) in enumerate(self.trees):
                if eps is None:
                    tree.fit(X, y, A, cat_idx, tbound)
                else:
                    tree.fit(X, y, A, cat_idx, tbound, eps/self.num_trees, b_med)

    # %% Predicting target values based on attributes
    # X: M samples and N features (M x N)
    # cat_idx: List of indices for categorical features
    # Returns predicted y
    def predict(self, X, cat_idx):
        M, N = np.shape(X)
        pred = np.zeros((M, self.num_trees))

        for i in range(self.num_trees):  # Prediction from each tree
            pred[:, i] = self.trees[i].predict(X, cat_idx)

        return np.mean(pred, axis=1)

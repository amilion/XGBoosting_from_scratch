import numpy as np
from typing import Sequence, Tuple, Self


class Node:
    """ Regression Tree Base Node class which splits itself recursively untill one of termination criterias been satisfied
    uses mean squared error as its loss function   
    """
    def __init__(
        self,
        X,
        y,
        min_records_to_split = 5,
        max_depth = 5,
    ):
        self.X = X
        self.y = y
        self.depth = max_depth
        self.stop_record = min_records_to_split
        self.node_value = self.y.mean()
        self.split_feature = None
        self.split_value = None
        self.lhs, self.rhs = self.split_self()
        self.epsilon = 1e-3
    
    def greedy_split(self):
        """
        sets a new value for self.feature which represents the feature on which we have splited our node and also
        sets a new value for self.value which represents the value on which our feature should be divided in to parts.
        """
        curr_gain = 0
        tree_mean = self.y.mean()
        G, H = Node.gradient(tree_mean, self.y), Node.hessian()
        feature_number = self.X.shape[1]
        for feature in range(feature_number):
            idxs = np.argsort(self.X[:, feature])
            G_L, H_L = 0, 0
            G_R, H_R = G, H           
            for row in range(self.X.shape[0]):
                l_actual, r_actual = self.y[idxs[:row+1]], self.y[idxs[row+1:]]
                G_L, H_L = Node.gradient(l_actual.mean(), l_actual), Node.hessian()
                G_R, H_R = Node.gradient(r_actual.mean(), r_actual), Node.hessian()
                new_gain = Node.gain(G_L, H_L, G_R, H_R)
                if (new_gain > curr_gain):
                    curr_gain = new_gain
                    self.split_feature = feature
                    self.split_value = self.X[:, feature][idxs[row]]                                           
        
    
    @staticmethod
    def loss_function(y_hat, y):
        """loss function
        Args:
            y_hat : predicts
            y : targets
        """
        return 0.5 * (y_hat - y)**2
    
    @staticmethod
    def gradient(output, y_true):
        """first order derivation of loss function: 2*(output - y_true)

        Args:
            output : output of the system
            y_true : actual values

        Returns:
            float : values of gradient
        """
        return np.sum(2 * (output - y_true))
    
    @staticmethod
    def hessian():
        """second order derivation of loss function: 2

        Returns:
            int: simply returns 2
        """
        return 2
    
    @staticmethod
    def impurity_function(grad, hess, _lambda = 1):
        """
            impurity function: eq (6) of the paper
            Args:
                grad : gradient
                hess : hessian 
                _lambda : hyperparameter to control the complexity
        """
        return grad**2 / (hess + _lambda)

    @staticmethod
    def gain(G_L, H_L, G_R, H_R, _gamma = 1):
        """
            calculates the information gain of the candidated split

        Args:
            G_L, G_R : left and right nodes gradients
            H_L, H_R : left and right nodes hessians
            _gamma : hyperparameter to control the complexity

        Returns:
            float: the gain
        """
        return Node.impurity_function(G_L, H_L) + Node.impurity_function(G_R, H_R) - Node.impurity_function(G_L + G_R, H_L + H_R) 
    
    def is_leaf(self) -> bool:
        """checks wether the self node is a leaf or not

        Returns:
            bool: returns true if self node is a leaf and false otherwise
        """
        if self.depth == 1 or self.X.shape[0] < self.stop_record:
            return True
        return False

    def split_self(self) -> Tuple[Self, Self] | Tuple[None, None]:
        """recursively splits the self node until one termination criteria satisfies

        Returns:
            Tuple[Self, Self] | Tuple[None, None]: returns left subtree and right subtree, returns None, None, if self node is a leaf
        """
        if self.is_leaf():
            return None, None
        self.greedy_split()
        left_idxs = self.X[:, self.split_feature] <= self.split_value
        right_idxs = ~left_idxs
        lhs = Node(self.X[left_idxs], self.y[left_idxs], max_depth=self.depth - 1)
        rhs = Node(self.X[right_idxs], self.y[right_idxs], max_depth=self.depth - 1)
        return lhs, rhs
    
    def _predict(self, X) -> float:
        """predicts the value for a single instance

        Args:
            X (Array) : input instance

        Returns:
            float: the estimated value for input instance
        """
        if self.is_leaf() or abs(self.split_value - X[self.split_feature]) <= self.epsilon:
            return self.node_value
        
        if (X[self.split_feature] <= self.split_value or (not self.rhs and self.lhs)):
            return self.lhs._predict(X)
        if (self.rhs):
            return self.rhs._predict(X)
        return self.node_value
        
    def predict(self, X) -> Sequence[float]:
        """predicts values of a Sequence of instances

        Args:
            X (Array): Sequence of instances

        Returns:
            Sequence[float]: output values for each instance respectively
        """
        return np.apply_along_axis(self._predict, axis=1, arr=X)


class GradientBoostingRegressor:
    """Source Paper is : XGBoost: A Scalable Tree Boosting System by Tianqi Chen and Carlos Guestrin
    """
    def __init__(self, number_of_ensembels = 5, max_depth = 3):
        self.ensamble_number = number_of_ensembels
        self.max_depth = max_depth
        self.ensambles = []
    
    def fit(self, X: Sequence[Sequence[float]], y: Sequence[float]):
        """fits the training instances according to their targets

        Args:
            X (Sequence[Sequence[float]]): values of predictors for training the ensemble
            y (Sequence[float]): values of targets for training the ensemble
        """
        base = y
        first_tree = Node(X=X, y=base)
        preds = first_tree.predict(X = X)
        self.ensambles.append(first_tree)
        for ensamble in range(self.ensamble_number - 1):
            base = base - preds
            T = Node(X=X, y=base, max_depth=self.max_depth)
            preds = T.predict(X = X)
            self.ensambles.append(T)
    
    def predict(self, X: Sequence[Sequence[float]]) -> float:
        """predicts the value for a given instance

        Args:
            X (Sequence[Sequence[float]]): input instance

        Returns:
            float: the estimated value for the given instance
        """
        res = 0
        for ensemble in self.ensambles:
            res += ensemble.predict(X)
        return res



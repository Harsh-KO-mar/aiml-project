import numpy as np
import pickle as pkl

class LDA:
    def __init__(self,k):
        self.n_components = k
        self.linear_discriminants = None

    def fit(self, X, y):
        """
        X: (n,d,d) array consisting of input features
        y: (n,) array consisting of labels
        return: Linear Discriminant np.array of size (d*d,k)
        """
        # TODO
        self.linear_discriminants=np.zeros((len(X[0]*len(X[0])),k)) # Modify as required 
        return(self.linear_discriminants)                 # Modify as required
        #END TODO 
    
    def transform(self, X, w):
        """
        w:Linear Discriminant array of size (d*d,1)
        return: np-array of the projected features of size (n,k)
        """
        # TODO
        projected=np.zeros((len(X),k))     # Modify as required
        return projected                   # Modify as required
        # END TODO

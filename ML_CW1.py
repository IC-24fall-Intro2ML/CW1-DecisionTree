import numpy as np
import matplotlib as plt

# Left nodes contain x <= than value

class stump:


    label = None
    feature = None
    value = None
    left = None
    right = None

    def __init__(self):

        pass

    def evaluate(self, x):
        if (self.feature is None):
            return self.label
        else:
            if (x[self.feature] <= self.value):
                return self.left.evaluate(x)
            else:
                return self.right.evaluate(x)
            


# X_y meaning X appended with y

def decision_tree_learning(X_y, d, max_d):
    if len(np.unique(X_y[:,-1])) == 1 or d>= max_d:
        new_stump = stump()
        #new_stump.label = X_y[:,-1][0]

        new_stump.label = np.bincount(X_y[:, -1].astype(int)).argmax()

        return new_stump, d
    
    else:
        feature, value = find_split(X_y)


        left_X, right_X = split(X_y, feature, value)

        left, l_d = decision_tree_learning(left_X, d+1, max_d)
        right, r_d = decision_tree_learning(right_X, d+1, max_d)

        new_stump = stump()
        new_stump.feature, new_stump.value, new_stump.left, new_stump.right= feature, value, left, right

        return new_stump, np.max([l_d, r_d])
 


def split(X_y, feature, value):
    return X_y[X_y[:, feature] <= value], X_y[X_y[:, feature] > value]




def find_split(X_y):

    # loop through all features, minus 1 for y
    best_ig = -np.inf

    best_split = (0,0)

    base_H = entropy(X_y)

    for f in range(0, X_y.shape[1] - 1):

        # loop through all available values of the features
        for v in X_y[:, f]:
            l, r = split(X_y, f, v)

            total_ct = X_y.shape[0]

            info_gain = base_H - (l.shape[0]/total_ct)*entropy(l) - (r.shape[0]/total_ct)*entropy(r)

            if info_gain > best_ig:
                best_ig = info_gain
                best_split = f, v

    return best_split







def entropy(X_y):
    H = 0
    for label in np.unique(X_y[:, -1]):
        p_label = np.sum(X_y[:,-1] == label)/(X_y[:,-1].shape[0])
        H -= p_label*np.log2(p_label)

    return H



def evaluate_accuracy(tree, X_test_y):

    correct_ct = 0

    X_test = X_test_y[:, :-1] 
    y_test = X_test_y[:, -1]   
    

    for i in range(X_test.shape[0]):

        y_hat = tree.evaluate(X_test[i])
        

        if y_hat == y_test[i]:
            correct_ct += 1
    

    acc = correct_ct / X_test.shape[0]
    
    return acc


clean_data = np.loadtxt("./noisy_dataset.txt")


tree, d = decision_tree_learning(clean_data, 0, 4)

print(evaluate_accuracy(tree, clean_data))
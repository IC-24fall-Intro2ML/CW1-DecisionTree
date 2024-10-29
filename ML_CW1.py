import numpy as np
import matplotlib.pyplot as plt

# Left nodes contain x <= than value


# stump = nodes, used interchangably here

class stump:


    label = None
    feature = None
    value = None
    left = None
    right = None

    leafcount = None

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

        binnedCount = np.bincount(X_y[:, -1].astype(int))

        new_stump.label = binnedCount.argmax()



        new_stump.leafcount = np.pad(binnedCount, (0, max(0, 5 - len(binnedCount))))

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


def find_prunable_nodes(node):
    
    prunable_nodes = []
    
    
    if node is None or node.feature is None:
        return prunable_nodes
    
    # Check if both children are leaf nodes
    if node.left and node.right:
        if node.left.feature is None and node.right.feature is None:
            # If both are leaf nodes, add the current node to the list
            prunable_nodes.append(node)

    # Recursively check the left and right subtrees
    prunable_nodes += find_prunable_nodes(node.left)
    prunable_nodes += find_prunable_nodes(node.right)

    return prunable_nodes
    
def pruneOneStump(s):

    totalBinnedCounts = s.left.leafcount + s.right.leafcount

    s.label = totalBinnedCounts.argmax()

    s.leafcount = totalBinnedCounts

    s.feature = None

    s.value = None

    s.left = None
    s.right = None


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



def pruneTree(tree, validationSet):

    baseAcc = evaluate_accuracy(tree, validationSet)

    improvement = True

    while(improvement):
        candidateNodes = find_prunable_nodes(tree)
        improvement = False
        

        for node in candidateNodes:

            # info for undoing the prune
            leftChild = node.left
            rightChild = node.right
            feature = node.feature
            value = node.value

            pruneOneStump(node)

            if (evaluate_accuracy(tree, validationSet) < baseAcc):

                # undo the prune if the accuracy got worse
                node.left = leftChild
                node.right = rightChild
                node.feature = feature
                node.value = value

                node.label = None
                node.leafcount = None
            else:
                improvement = True
                break




def kfoldCV(k, X_y, maxd):

    accs = np.zeros(k)

    validationSize = X_y.shape[0]//k

    shuffledX_y = np.random.permutation(X_y)

    for i in range(k):

        start = (i) * validationSize
        end = (i + 1) * validationSize

        validationSet = shuffledX_y[start:end,:]

        trainingSet = np.delete(shuffledX_y, np.s_[start:end], axis=0)

        tree, d = decision_tree_learning(trainingSet, 0, maxd)

        accs[i] = evaluate_accuracy(tree, validationSet)

    return accs, np.average(accs)


def confusionMatrix(y, y_hat, cat_count):
    #cat_count = np.unique(np.concatenate(y, y_hat)).shape[0]

    conf = np.zeros(cat_count, cat_count)

    for Y, Yhat in zip(y, y_hat):
        
        conf[Y, Yhat] += 1

    return conf


clean_data = np.loadtxt("./clean_dataset.txt")


tree, d = decision_tree_learning(clean_data, 0, 10)


def visualize_tree(node, depth=0, x=0.5, y=1.0, x_offset=0.3, ax=None):

    if ax is None:
        plt.close('all') 
        fig, ax = plt.subplots(figsize=(100, 80)) 
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_axis_off()


    if node is None:
        return

   
    if node.feature is None:
        ax.text(x, y, f'Label: {node.label}', bbox=dict(facecolor='white', edgecolor='black'), ha='center')
    else:
        ax.text(x, y, f'Feature: {node.feature}\n<= {node.value}', bbox=dict(facecolor='white', edgecolor='black'), ha='center')

   
    next_y = y - 0.1  # Move downwards for child
    next_x_offset = x_offset / (depth + 0.3)  # Reduce x_offset as depth increases



    if node.left:
        next_x_left = x - next_x_offset  # Move left child to the left
        ax.plot([x, next_x_left], [y, next_y], 'k-')  # Draw line to the left child
        visualize_tree(node.left, depth + 1, next_x_left, next_y, x_offset, ax)


    if node.right:
        next_x_right = x + next_x_offset  
        ax.plot([x, next_x_right], [y, next_y], 'k-')  
        visualize_tree(node.right, depth + 1, next_x_right, next_y, x_offset, ax)

    if depth == 0:
        plt.show()



visualize_tree(tree, depth=0, x=0.5, y=1.0, x_offset=0.1, ax=None)

#print(kfoldCV(10, clean_data, 5))

print(evaluate_accuracy(tree, clean_data))


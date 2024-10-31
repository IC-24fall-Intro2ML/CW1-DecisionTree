import numpy as np
import matplotlib.pyplot as plt

# Left nodes are less than or equal to the value, right nodes are greater than the value


# stump = nodes, used interchangably here

class stump:
    """
    A class to represent a decision tree stump.

    Args:
        label: The label of the stump.
        feature: The feature to split on.
        value: The value to split on.
        left: The left child stump.
        right: The right child stump.
    """

    leafcount = None  # count of each label in the leaf node

    def __init__(self, label=None, feature=None, value=None, left=None, right=None):
        self.label = label
        self.feature = feature
        self.value = value
        self.left = left
        self.right = right

    def evaluate(self, x):
        """
        Recursively evaluate the stump on a given input.

        Args:
            x: The input to evaluate the stump on.

        Returns:
            The label of the stump.
        """
        if (self.feature is None):
            return self.label
        else:
            if (x[self.feature] <= self.value):
                return self.left.evaluate(x)
            else:
                return self.right.evaluate(x)


def decision_tree_learning(train_data, d, max_d=np.inf):
    """
    Recursively build a decision tree.

    Args:
        train_data: The training data.
        d: The current depth of the tree.
        max_d: The maximum depth of the tree. The default value is positive infinity, meaning the depth of the tree is not limited.

    Returns:
        new_stump: The root stump of the decision tree.
        d: The depth of the decision tree.
    """
    # Base case: If the node is pure or the maximum depth is reached
    if len(np.unique(train_data[:, -1])) == 1 or d >= max_d:
        new_stump = stump()

        binnedCount = np.bincount(train_data[:, -1].astype(int))

        new_stump.label = binnedCount.argmax()  # most common label

        new_stump.leafcount = np.pad(
            binnedCount, (0, max(0, 5 - len(binnedCount))))

        return new_stump, d

    # Recursive case: Find the best split and split the data
    feature, value = find_split(train_data)

    left_X, right_X = split(train_data, feature, value)

    left, l_d = decision_tree_learning(left_X, d + 1, max_d)
    right, r_d = decision_tree_learning(right_X, d + 1, max_d)

    # Create a new stump with split information
    new_stump = stump()
    new_stump.feature, new_stump.value, new_stump.left, new_stump.right = feature, value, left, right

    return new_stump, np.max([l_d, r_d])


def split(train_data, feature, value):
    """
    Split the data based on a feature and value.

    Args:
        train_data: The training data.
        feature: The feature to split on.
        value: The value to split on.

    Returns:
        left: The left split of the data.
        right: The right split of the data.
    """
    left = train_data[train_data[:, feature] <= value]
    right = train_data[train_data[:, feature] > value]
    return left, right


def find_split(train_data):
    """
    Find the best split for the data.

    Args:
        train_data: The training data.

    Returns:
        best_split: The best split for the data.
    """
    best_ig = -np.inf
    best_split = (0, 0)
    base_H = entropy(train_data)

    # Iterate through all features and values to find the best split
    for f in range(train_data.shape[1] - 1):
        for v in train_data[:, f]:
            # Split the data based on the feature and value
            left, right = split(train_data, f, v)

            total_ct = train_data.shape[0]

            info_gain = base_H - \
                (left.shape[0] / total_ct) * entropy(left) - \
                (right.shape[0] / total_ct) * entropy(right)

            if info_gain > best_ig:
                best_ig = info_gain
                best_split = f, v

    return best_split


def entropy(data):
    """
    Calculate the entropy of the data.

    Args:
        data: The data to calculate the entropy of.

    Returns:
        H: The entropy of the data.
    """
    H = 0
    for label in np.unique(data[:, -1]):
        p_label = np.sum(data[:, -1] == label) / (data[:, -1].shape[0])
        H -= p_label * np.log2(p_label)

    return H


def find_prunable_nodes(node):
    """
    Find node elgible for pruning.

    Args:
        node: The node to check for pruning.

    Returns:
        prunable_nodes: The list of nodes eligible for pruning.
    """
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


def pruneOneStump(node):
    """
    Prune and merge the leaf children of one node.

    Args:
        node: The node to prune.
    """
    totalBinnedCounts = node.left.leafcount + node.right.leafcount

    node.label = totalBinnedCounts.argmax()
    node.leafcount = totalBinnedCounts
    node.feature = None
    node.value = None
    node.left = None
    node.right = None


def evaluate_accuracy(tree, X_test_y):
    """
    Evaluate the accuracy of the decision tree on a test set.

    Args:
        tree: The decision tree.
        X_test_y: The test set.

    Returns:
        acc: The accuracy of the decision tree on the test set.
    """
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
    """
    Prune the decision tree using the validation set.

    Args:
        tree: The decision tree.
        validationSet: The validation set.
    """
    baseAcc = evaluate_accuracy(tree, validationSet)

    improvement = True

    while (improvement):
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


def kfoldCV(k, data, maxd):
    """
    Compute the k-fold cross validation metrics.

    Args:
        k: The number of folds.
        data: The data to evaluate.
        maxd: The maximum depth of the decision tree.

    Returns:
        A tuple containing:
            avg_confusion: The average confusion matrix.
            avg_accuracy: The average accuracy.
            precisions: The precisions for each class.
            recalls: The recalls for each class.
            f1_scores: The F1 scores for each class.
    """
    cat_count = len(np.unique(data[:, -1]))
    confusions = np.zeros((cat_count, cat_count))
    validationSize = data.shape[0] // k
    shuffled_data = np.random.permutation(data)

    for i in range(k):
        start = i * validationSize
        end = (i + 1) * validationSize

        validationSet = shuffled_data[start:end, :]
        trainingSet = np.delete(shuffled_data, np.s_[start:end], axis=0)

        tree, d = decision_tree_learning(trainingSet, 0, maxd)

        X_test = validationSet[:, :-1]
        y_test = validationSet[:, -1]
        y_hat = np.array([tree.evaluate(x) for x in X_test])

        confusion_matrix = confusionMatrix(
            y_test.astype(int), y_hat.astype(int), cat_count)
        confusions += confusion_matrix

    avg_confusion = confusions / k

    avg_accuracy = np.sum(np.diag(avg_confusion)) / np.sum(avg_confusion)

    precisions, recalls, f1_scores = [], [], []
    for i in range(cat_count):
        tp = avg_confusion[i, i]
        fn = np.sum(avg_confusion[i, :]) - tp
        fp = np.sum(avg_confusion[:, i]) - tp

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = (2 * precision * recall) / (precision +
                                         recall) if (precision + recall) > 0 else 0

        recalls.append(recall)
        precisions.append(precision)
        f1_scores.append(f1)

    precisions = [float(p) for p in precisions]
    recalls = [float(r) for r in recalls]
    f1_scores = [float(f) for f in f1_scores]

    return tree, avg_confusion, avg_accuracy, precisions, recalls, f1_scores


def pruneTest(vsize, data, maxd):
    """
    Prune the decision tree using the validation set.

    Args:
        vsize: The size of the validation set.
        data: The data to evaluate.
        maxd: The maximum depth of the decision tree.

    Returns:
        train_acc: The accuracy of the decision tree on the training set.
        pre_acc: The accuracy of the decision tree on the validation set before pruning.
        post_acc: The accuracy of the decision tree on the validation set after pruning.
    """
    validationSize = int(data.shape[0] * vsize)

    shuffled_data = np.random.permutation(data)

    start = 0
    end = int(validationSize)

    validationSet = shuffled_data[start:end, :]

    trainingSet = np.delete(shuffled_data, np.s_[start:end], axis=0)

    tree, d = decision_tree_learning(trainingSet, 0, maxd)

    pre_acc = evaluate_accuracy(tree, validationSet)

    pruneTree(tree, validationSet)

    post_acc = evaluate_accuracy(tree, validationSet)

    train_acc = evaluate_accuracy(tree, trainingSet)

    return train_acc, pre_acc, post_acc


def confusionMatrix(y, y_hat, cat_count):
    """
    Compute the confusion matrix.

    Args:
        y: The true labels.
        y_hat: The predicted labels.
        cat_count: The number of categories.

    Returns:
        conf: The confusion matrix.
    """
    conf = np.zeros((cat_count, cat_count))
    for Y, Yhat in zip(y, y_hat):
        conf[Y-1, Yhat-1] += 1

    return conf


def findAvgDepth(tree, depth=0):
    """
    Find the average depth of the leaf nodes in the decision tree.

    Args:
        tree: The decision tree.
        depth: The current depth of the tree.

    Returns:
        totalDepth: The total depth of the leaf nodes.
        totalLeafCount: The total number of leaf nodes.
    """
    if tree.feature is None:
        return (depth, 1)

    leftDepthSum, leftLeafCount = findAvgDepth(tree.left, depth + 1)
    rightDepthSum, rightLeafCount = findAvgDepth(tree.right, depth + 1)

    totalDepth = leftDepthSum + rightDepthSum
    totalLeafCount = leftLeafCount + rightLeafCount

    return totalDepth, totalLeafCount


def calculateAvgDepth(tree):
    """
    Calculate the average depth of the leaf nodes in the decision tree.

    Args:
        tree: The decision tree.

    Returns:
        The average depth of the leaf nodes in the decision tree.
    """
    totalDepth, totalLeafCount = findAvgDepth(tree)
    if totalLeafCount == 0:
        return 0
    return totalDepth / totalLeafCount


def visualize_tree(node, depth=0, x=0.5, y=1.0, x_offset=0.3, ax=None):
    """
    Visualize the decision tree.

    Args:
        node: The node to visualize.
        depth: The current depth of the tree.
        x: The x-coordinate of the node.
        y: The y-coordinate of the node.
        x_offset: The x-offset of the node.
        ax: The axis to plot on.
    """

    if ax is None:
        plt.close('all')
        fig, ax = plt.subplots(figsize=(100, 80))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_axis_off()

    if node is None:
        return

    if node.feature is None:
        ax.text(x, y, f'Label: {node.label}', bbox=dict(
            facecolor='white', edgecolor='black'), ha='center')
    else:
        ax.text(x, y, f'Feature: {node.feature}\n<= {node.value}', bbox=dict(
            facecolor='white', edgecolor='black'), ha='center')

    next_y = y - 0.1  # Move downwards for child
    # Reduce x_offset as depth increases
    next_x_offset = x_offset / (depth + 0.3)

    if node.left:
        next_x_left = x - next_x_offset  # Move left child to the left
        # Draw line to the left child
        ax.plot([x, next_x_left], [y, next_y], 'k-')
        visualize_tree(node.left, depth + 1, next_x_left, next_y, x_offset, ax)

    if node.right:
        next_x_right = x + next_x_offset
        ax.plot([x, next_x_right], [y, next_y], 'k-')
        visualize_tree(node.right, depth + 1,
                       next_x_right, next_y, x_offset, ax)

    if depth == 0:
        plt.show()


def load_data(file_path):
    """
    Load the data from a file.

    Args:
        file_path: The path to the file.

    Returns:
        data: The data from the file.
    """
    data = np.loadtxt(file_path)
    return data


def print_info(confusion=None, accuracy=None, precisions=None, recalls=None, f1_scores=None):
    """
    Print the classification metrics.

    Args:
        confusion: The confusion matrix.
        accuracy: The accuracy.
        precisions: The precisions.
        recalls: The recalls.
        f1_scores: The F1 scores.
    """
    if confusion is not None:
        print("confusion: ", confusion)
    if accuracy is not None:
        print("accuracy: ", round(accuracy, 6))
    if precisions is not None:
        print("precisions: ", [round(p, 6) for p in precisions])
    if recalls is not None:
        print("recalls: ", [round(r, 6) for r in recalls])
    if f1_scores is not None:
        print("F1-measures: ", [round(f, 6) for f in f1_scores])


clean_data = load_data("./wifi_db/clean_dataset.txt")
noisy_data = load_data("./wifi_db/noisy_dataset.txt")

# Train the decision tree and evaluate the accuracy using clean data
print("Cross validation classification metrics for clean data: ")
tree, confusion, accuracy, precisions, recalls, f1_scores = kfoldCV(
    10, clean_data, 20)
print_info(confusion, accuracy, precisions, recalls, f1_scores)
visualize_tree(tree)

print("-" * 50)

# Train the decision tree and evaluate the accuracy using noisy data
print("Cross validation classification metrics for noisy data: ")
tree, confusion, accuracy, precisions, recalls, f1_scores = kfoldCV(
    10, noisy_data, 20)
print_info(confusion, accuracy, precisions, recalls, f1_scores)
visualize_tree(tree)

# print(evaluate_accuracy(tree, clean_data))

print("-" * 50)

# Prune the tree and evaluate the accuracy using clean data
train_acc, pre_acc, post_acc = pruneTest(0.25, clean_data, 25)
print("Train accuracy: ", train_acc)
print("Pre-prune accuracy: ", pre_acc)
print("Post-prune accuracy: ", post_acc)

print("-" * 50)

# Prune the tree and evaluate the accuracy using noisy data
train_acc, pre_acc, post_acc = pruneTest(0.25, noisy_data, 25)
print("Train accuracy: ", train_acc)
print("Pre-prune accuracy: ", pre_acc)
print("Post-prune accuracy: ", post_acc)


# tree, d = decision_tree_learning(clean_data, 0, 10)

# visualize_tree(tree, depth=0, x=0.5, y=1.0, x_offset=0.1, ax=None)

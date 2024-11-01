import numpy as np
import matplotlib.pyplot as plt


class Stump:
    """
    A class to represent a decision tree stump (node).
    """

    def __init__(self, label=None, feature=None, value=None, left=None, right=None):
        self.label = label
        self.feature = feature
        self.value = value
        self.left = left
        self.right = right
        self.leafcount = None

    def evaluate(self, x):
        if self.feature is None:
            return self.label
        elif x[self.feature] <= self.value:
            return self.left.evaluate(x)
        else:
            return self.right.evaluate(x)


class DecisionTreeClassifier:
    """
    A class to represent a decision tree classifier.
    """

    def __init__(self, max_depth=np.inf):
        self.max_depth = max_depth
        self.tree = None

    def decision_tree_learning(self, train_data, d=0, max_d=np.inf):
        """
        Recursively builds a decision tree.

        Args:
            train_data: The training data.
            d: The current depth of the decision tree.
            max_d: The maximum depth of the decision tree.

        Returns:
            new_stump: The decision tree stump.
            d: The depth of the decision tree.
        """
        if len(np.unique(train_data[:, -1])) == 1 or d >= self.max_depth:
            new_stump = Stump()
            binned_count = np.bincount(train_data[:, -1].astype(int))
            new_stump.label = binned_count.argmax()
            new_stump.leafcount = np.pad(
                binned_count, (0, max(0, 5 - len(binned_count))))
            return new_stump, d

        feature, value = self.find_split(train_data)
        left_X, right_X = self.split(train_data, feature, value)
        left, l_d = self.decision_tree_learning(left_X, d + 1)
        right, r_d = self.decision_tree_learning(right_X, d + 1)

        new_stump = Stump(feature=feature, value=value, left=left, right=right)
        return new_stump, max(l_d, r_d)

    def split(self, train_data, feature, value):
        left = train_data[train_data[:, feature] <= value]
        right = train_data[train_data[:, feature] > value]
        return left, right

    def find_split(self, train_data):
        """
        Finds the best split for the decision tree.

        Args:
            train_data: The training data.

        Returns:
            best_split: The best split for the decision tree.
        """
        best_ig = -np.inf
        best_split = (0, 0)
        base_H = self.entropy(train_data)
        for f in range(train_data.shape[1] - 1):
            for v in train_data[:, f]:
                left, right = self.split(train_data, f, v)
                total_ct = train_data.shape[0]
                info_gain = base_H - (left.shape[0] / total_ct) * self.entropy(left) - \
                    (right.shape[0] / total_ct) * self.entropy(right)
                if info_gain > best_ig:
                    best_ig = info_gain
                    best_split = (f, v)
        return best_split

    def entropy(self, data):
        H = 0
        for label in np.unique(data[:, -1]):
            p_label = np.sum(data[:, -1] == label) / data[:, -1].shape[0]
            H -= p_label * np.log2(p_label)
        return H

    def find_prunable_nodes(self, node):
        """
        Finds the prunable nodes in the decision tree.

        Args:
            node: The decision tree node.

        Returns:
            prunable_nodes: The prunable nodes in the decision tree.
        """
        prunable_nodes = []
        if node is None or node.feature is None:
            return prunable_nodes
        if node.left and node.right:
            if node.left.feature is None and node.right.feature is None:
                prunable_nodes.append(node)
        prunable_nodes.extend(self.find_prunable_nodes(node.left))
        prunable_nodes.extend(self.find_prunable_nodes(node.right))
        return prunable_nodes

    def prune_one_stump(self, node):
        total_binned_counts = node.left.leafcount + node.right.leafcount
        node.label = total_binned_counts.argmax()
        node.leafcount = total_binned_counts
        node.feature = None
        node.value = None
        node.left = None
        node.right = None

    def prune_tree(self, tree, validation_set):
        """
        Prunes the decision tree using the validation set.

        Args:
            tree: The decision tree.
            validation_set: The validation set.
        """
        base_acc = self.val_error(tree, validation_set)
        improvement = True
        while improvement:
            candidate_nodes = self.find_prunable_nodes(tree)
            improvement = False
            for node in candidate_nodes:
                left, right, feature, value = node.left, node.right, node.feature, node.value
                self.prune_one_stump(node)
                if self.val_error(tree, validation_set) < base_acc:
                    node.left, node.right, node.feature, node.value = left, right, feature, value
                    node.label = None
                    node.leafcount = None
                else:
                    improvement = True
                    break

    def prune_test(self, vsize, data, maxd):
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

        tree, d = self.decision_tree_learning(trainingSet, 0, maxd)

        pre_acc = self.val_error(tree, validationSet)

        self.prune_tree(tree, validationSet)

        post_acc = self.val_error(tree, validationSet)

        train_acc = self.val_error(tree, trainingSet)

        return train_acc, pre_acc, post_acc

    def val_error(self, tree, X_test_y):
        correct_ct = 0
        X_test, y_test = X_test_y[:, :-1], X_test_y[:, -1]
        for i in range(X_test.shape[0]):
            y_hat = tree.evaluate(X_test[i])
            if y_hat == y_test[i]:
                correct_ct += 1
        return correct_ct / X_test.shape[0]

    def kfold_cross_validation(self, k, data):
        cat_count = len(np.unique(data[:, -1]))
        validation_size = data.shape[0] // k
        shuffled_data = np.random.permutation(data)

        # To accumulate confusion matrices of every fold
        confusions = np.zeros((cat_count, cat_count))

        for i in range(k):
            start, end = i * validation_size, (i + 1) * validation_size
            validation_set = shuffled_data[start:end, :]
            training_set = np.delete(shuffled_data, np.s_[start:end], axis=0)

            self.tree, _ = self.decision_tree_learning(training_set)

            # Evaluate metrics on the validation set for the current fold
            confusion_matrix, _, _, _, _ = self.evaluate_metrics(
                self.tree, validation_set)

            # Accumulate confusion matrix across folds
            confusions += confusion_matrix

        avg_confusion = confusions / k
        avg_accuracy, avg_recalls, avg_precisions, avg_f1_scores = self.calculate_metrics_from_confusion_matrix(
            avg_confusion)

        return avg_confusion, avg_accuracy, avg_recalls, avg_precisions, avg_f1_scores

    def nested_cross_validation(self, data, outer_folds=10, inner_folds=9):
        """
        Performs nested cross-validation on the data, evaluating using confusion matrix, accuracy, precision, recall, and F1-score.

        Args:
            data: The data to evaluate.
            outer_folds: The number of outer folds.
            inner_folds: The number of inner folds.

        Returns:
            avg_confusion: Average confusion matrix across all outer folds.
            avg_accuracy: Average accuracy across all outer folds.
            avg_recalls: Average recall across all outer folds.
            avg_precisions: Average precision across all outer folds.
            avg_f1_scores: Average F1-score across all outer folds.
        """
        outer_validation_size = data.shape[0] // outer_folds
        shuffled_data = np.random.permutation(data)

        # To accumulate metrics across outer folds
        confusion_matrices = []

        for p in range(outer_folds):
            start, end = p * \
                outer_validation_size, (p + 1) * outer_validation_size
            outer_test_set = shuffled_data[start:end, :]
            outer_train_val_set = np.delete(
                shuffled_data, np.s_[start:end], axis=0)

            inner_validation_size = outer_train_val_set.shape[0] // inner_folds
            inner_results = []

            for k in range(inner_folds):
                inner_start, inner_end = k * \
                    inner_validation_size, (k + 1) * inner_validation_size
                inner_validation_set = outer_train_val_set[inner_start:inner_end, :]
                inner_training_set = np.delete(outer_train_val_set, np.s_[
                    inner_start:inner_end], axis=0)

                # Train on inner training set
                tree, _ = self.decision_tree_learning(inner_training_set)

                # Prune on inner validation set
                self.prune_tree(tree, inner_validation_set)

                # Evaluate pruned tree on outer test set
                confusion_matrix, _, _, _, _ = self.evaluate_metrics(
                    tree, outer_test_set)
                inner_results.append(confusion_matrix)

            # Average results from inner folds for the current outer test fold
            avg_inner_confusion = np.mean(inner_results, axis=0)

            # Store results across outer folds
            confusion_matrices.append(avg_inner_confusion)

            # print(f"Outer Fold {p + 1}/{outer_folds} - Test Fold Metrics: ")
            # print(f"Accuracy: {avg_inner_accuracy:.4f}, ")
            # print(f"Recall per class: {avg_inner_recalls}, ")
            # print(f"Precision per class: {avg_inner_precisions}, ")
            # print(f"F1 per class: {avg_inner_f1s}")

        # Final average metrics across all outer folds
        avg_confusion = np.mean(confusion_matrices, axis=0)

        avg_accuracy, avg_recalls, avg_precisions, avg_f1s = self.calculate_metrics_from_confusion_matrix(
            avg_confusion)

        return avg_confusion, avg_accuracy, avg_recalls, avg_precisions, avg_f1s

    def confusion_matrix(self, y, y_hat, cat_count):
        conf = np.zeros((cat_count, cat_count))
        for Y, Yhat in zip(y, y_hat):
            conf[Y-1, Yhat-1] += 1
        return conf

    def calculate_metrics_from_confusion_matrix(self, confusion_matrix):
        """
        Calculate accuracy, recall, precision, and F1 scores from a confusion matrix.

        Args:
            confusion_matrix (np.array): Confusion matrix of shape (n_classes, n_classes).

        Returns:
            accuracy (float): Overall accuracy.
            recalls (list): Recall per class.
            precisions (list): Precision per class.
            f1_scores (list): F1 score per class.
        """
        accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)

        recalls, precisions, f1_scores = [], [], []
        for i in range(confusion_matrix.shape[0]):
            tp = confusion_matrix[i, i]
            fn = np.sum(confusion_matrix[i, :]) - tp
            fp = np.sum(confusion_matrix[:, i]) - tp
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = (2 * precision * recall) / (precision +
                                             recall) if (precision + recall) > 0 else 0
            recalls.append(recall)
            precisions.append(precision)
            f1_scores.append(f1)

        return accuracy, recalls, precisions, f1_scores

    def evaluate_metrics(self, tree, X_test_y):
        """
        Evaluate the confusion matrix, accuracy, recalls, precisions, and F1-scores of the decision tree.

        Args:
            tree: The decision tree to evaluate.
            X_test_y: The test set (features and labels).

        Returns:
            confusion_matrix: Confusion matrix.
            accuracy: Accuracy of the model on the test set.
            recalls: Recall for each class.
            precisions: Precision for each class.
            f1-scores: F1-score for each class.
        """
        X_test, y_test = X_test_y[:, :-1], X_test_y[:, -1]
        y_hat = np.array([tree.evaluate(x) for x in X_test])

        # Confusion matrix calculation
        cat_count = len(np.unique(y_test))
        confusion_matrix = self.confusion_matrix(
            y_test.astype(int), y_hat.astype(int), cat_count)

        accuracy, recalls, precisions, f1_scores = self.calculate_metrics_from_confusion_matrix(
            confusion_matrix)

        return confusion_matrix, accuracy, recalls, precisions, f1_scores

    def find_avg_depth(self, node, depth=0):
        if node.feature is None:  # Node is a leaf
            return depth, 1
        left_depth_sum, left_leaf_count = self.find_avg_depth(
            node.left, depth + 1) if node.left else (0, 0)
        right_depth_sum, right_leaf_count = self.find_avg_depth(
            node.right, depth + 1) if node.right else (0, 0)
        total_depth = left_depth_sum + right_depth_sum
        total_leaf_count = left_leaf_count + right_leaf_count
        return total_depth, total_leaf_count

    def calculate_avg_depth(self):
        total_depth, total_leaf_count = self.find_avg_depth(self.tree)
        return total_depth / total_leaf_count if total_leaf_count > 0 else 0

    def visualize_tree(self, node=None, depth=0, x=0.5, y=1.0, x_offset=0.3, ax=None):
        if node is None:
            node = self.tree
        if ax is None:
            plt.close('all')
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_axis_off()
        if node.feature is None:
            ax.text(x, y, f'Label: {node.label}', bbox=dict(
                facecolor='white', edgecolor='black'), ha='center')
        else:
            ax.text(x, y, f'Feature: {node.feature}\n<= {node.value}', bbox=dict(
                facecolor='white', edgecolor='black'), ha='center')
        next_y = y - 0.1
        next_x_offset = x_offset / (depth + 1)
        if node.left:
            next_x_left = x - next_x_offset
            ax.plot([x, next_x_left], [y, next_y], 'k-')
            self.visualize_tree(node.left, depth + 1,
                                next_x_left, next_y, x_offset, ax)
        if node.right:
            next_x_right = x + next_x_offset
            ax.plot([x, next_x_right], [y, next_y], 'k-')
            self.visualize_tree(node.right, depth + 1,
                                next_x_right, next_y, x_offset, ax)
        if depth == 0:
            plt.show()


def load_data(file_path):
    return np.loadtxt(file_path)


def print_info(confusion=None, accuracy=None, recalls=None, precisions=None, f1_scores=None):
    if confusion is not None:
        print("Confusion Matrix:\n", confusion)
    if accuracy is not None:
        print("Accuracy:", round(accuracy, 4))
    if recalls is not None:
        print("Recalls:", list(map(float, [round(r, 4) for r in recalls])))
    if precisions is not None:
        print("Precisions:", list(
            map(float, [round(p, 4) for p in precisions])))
    if f1_scores is not None:
        print("F1-Scores:", list(map(float, [round(f, 4) for f in f1_scores])))


# def calculate_metrics_from_confusion_matrix(confusion_matrix):
    """
    Calculate accuracy, recall, precision, and F1 scores from a confusion matrix.

    Args:
        confusion_matrix (np.array): Confusion matrix of shape (n_classes, n_classes).

    Returns:
        accuracy (float): Overall accuracy.
        recalls (list): Recall per class.
        precisions (list): Precision per class.
        f1_scores (list): F1 score per class.
    """
    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)

    recalls, precisions, f1_scores = [], [], []
    for i in range(confusion_matrix.shape[0]):
        tp = confusion_matrix[i, i]
        fn = np.sum(confusion_matrix[i, :]) - tp
        fp = np.sum(confusion_matrix[:, i]) - tp
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = (2 * precision * recall) / (precision +
                                         recall) if (precision + recall) > 0 else 0
        recalls.append(recall)
        precisions.append(precision)
        f1_scores.append(f1)

    return accuracy, recalls, precisions, f1_scores


# Main script
if __name__ == "__main__":
    clean_data = load_data("./wifi_db/clean_dataset.txt")
    noisy_data = load_data("./wifi_db/noisy_dataset.txt")

    dt_clean = DecisionTreeClassifier(max_depth=np.inf)
    dt_noisy = DecisionTreeClassifier(max_depth=np.inf)

    np.set_printoptions(suppress=True)

    print("Cross-validation metrics for clean data:")
    confusion, accuracy, recalls, precisions, f1_scores = dt_clean.kfold_cross_validation(
        10, clean_data)
    clean_avg_depth = dt_clean.calculate_avg_depth()
    print(f"Average Depth: {round(clean_avg_depth, 2)}")
    print_info(confusion, accuracy, recalls, precisions, f1_scores)

    dt_clean.visualize_tree()

    print("\n" + "-" * 50 + "\n")

    print("Cross-validation metrics for noisy data:")
    confusion, accuracy, recalls, precisions, f1_scores = dt_noisy.kfold_cross_validation(
        10, noisy_data)
    noisy_avg_depth = dt_noisy.calculate_avg_depth()
    print(f"Average Depth: {round(noisy_avg_depth, 2)}")
    print_info(confusion, accuracy, recalls, precisions, f1_scores)

    dt_noisy.visualize_tree()

    print("\n" + "-" * 50 + "\n")

    # Nested cross-validation for clean data
    print("Nested Cross-Validation for Clean Data:")
    outer_folds = 10
    print(f"Average Metrics across {outer_folds} outer folds:")
    confusion, accuracy, recalls, precisions, f1_scores = dt_clean.nested_cross_validation(
        clean_data, outer_folds=10, inner_folds=9)
    print_info(confusion, accuracy, recalls, precisions, f1_scores)

    print("\n" + "-" * 50 + "\n")

    # Nested cross-validation for noisy data
    print("Nested Cross-Validation for Noisy Data:")
    outer_folds = 10
    print(f"Average Metrics across {outer_folds} outer folds:")
    confusion, accuracy, recalls, precisions, f1_scores = dt_noisy.nested_cross_validation(
        noisy_data, outer_folds=10, inner_folds=9)
    print_info(confusion, accuracy, recalls, precisions, f1_scores)

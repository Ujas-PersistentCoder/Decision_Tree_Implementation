"""
This module contains the main Node and DecisionTree class definitions.
It orchestrates the tree-building (fit) and prediction processes.
"""
import pandas as pd
# Use a relative import to bring in functions from the same package
from .utils import *

class Node:
    """
    A class representing a single node in the Decision Tree.
    A node can either be a decision node with a split condition
    or a leaf node with a prediction value.
    """
    def __init__(self, value=None):
        # Value for a leaf node (the prediction)
        self.value = value
        
        # Attributes for a decision node
        self.feature = None
        self.threshold = None
        self.is_real_feature = False # Tracks if the split is on a real or discrete feature
        self.left_child = None
        self.right_child = None

class DecisionTree:
    """
    A Decision Tree classifier and regressor.

    This class implements a decision tree algorithm from scratch, capable of handling
    both classification and regression tasks with real or discrete features.

    Attributes:
        criterion (str): The function to measure the quality of a split.
                         Supported criteria are "gini_index" and "entropy".
        max_depth (int): The maximum depth the tree is allowed to grow.
    """
    def __init__(self, criterion: str = "entropy", max_depth: int = 5):
        """Initializes the DecisionTree."""
        if criterion not in ["entropy", "gini_index"]:
            raise ValueError("Criterion must be 'entropy' or 'gini_index'")
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = None

    def _grow_tree(self, X, y, parent_val, depth=0):
        """
        Recursively grows the decision tree by finding the best split at each node.
        This is the core training logic.
        """
        # If a node becomes empty, predict the majority value of its parent
        if len(y) == 0:
            return Node(value=parent_val)

        # Determine the prediction for the current node if it were a leaf
        current_val = y.mode()[0] if not check_ifreal(y) else y.mean()

        # Base cases for stopping recursion (creating a leaf node)
        if depth >= self.max_depth or y.nunique() == 1 or X.shape[0] < 2:
            return Node(value=current_val)
        
        features = list(X.columns)
        best_feature, best_value = opt_split_attribute(X, y, self.criterion, features)

        # If no split provides any information gain, create a leaf
        if best_feature is None:
            return Node(value=current_val)

        # Create a new decision node
        node = Node()
        node.feature = best_feature
        node.threshold = best_value
        node.is_real_feature = check_ifreal(X[best_feature])

        # Split the data and grow the children
        X_left, y_left, X_right, y_right = split_data(X, y, best_feature, best_value)
        node.left_child = self._grow_tree(X_left, y_left, current_val, depth + 1)
        node.right_child = self._grow_tree(X_right, y_right, current_val, depth + 1)
        return node

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Builds the decision tree from a training set (X, y).

        Args:
            X (pd.DataFrame): The training input samples.
            y (pd.Series): The target values.
        """
        # Calculate the root's majority value for handling empty leaf nodes
        parent_val = y.mode()[0] if not check_ifreal(y) else y.mean()
        self.root = self._grow_tree(X, y, parent_val)

    def _traverse_tree(self, row, node):
        """
        Recursively traverses the tree to find a prediction for a single data row.
        """
        if node.value is not None:
            return node.value

        feature_value = row[node.feature]
        if node.is_real_feature:
            if feature_value <= node.threshold:
                return self._traverse_tree(row, node.left_child)
            else:
                return self._traverse_tree(row, node.right_child)
        else: # Discrete feature
            if feature_value == node.threshold:
                return self._traverse_tree(row, node.left_child)
            else:
                return self._traverse_tree(row, node.right_child)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predicts the class or value for each sample in X.

        Args:
            X (pd.DataFrame): The input samples to predict.

        Returns:
            pd.Series: A Series containing the predicted values.
        """
        return X.apply(lambda row: self._traverse_tree(row, self.root), axis=1)

    def _plot_recursive(self, node, indent=""):
        """Helper function to recursively print the tree structure."""
        if node.value is not None:
            print(indent + "--> Predict:", node.value)
            return

        op = "<=" if node.is_real_feature else "=="
        question = f"? ({node.feature} {op} {node.threshold})"
        print(indent + question)

        print(indent + "Y:")
        self._plot_recursive(node.left_child, indent + "    ")
        print(indent + "N:")
        self._plot_recursive(node.right_child, indent + "    ")

    def plot(self):
        """
        Prints a text-based visualization of the decision tree.
        """
        self._plot_recursive(self.root)

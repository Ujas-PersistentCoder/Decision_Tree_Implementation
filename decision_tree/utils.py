"""
utils.py

This module contains the core mathematical and helper functions used by the
DecisionTree class to build the tree. This includes functions for calculating
impurity, information gain, and finding the optimal feature split.
"""
import numpy as np
import pandas as pd

def check_ifreal(y: pd.Series) -> bool:
    """
    Heuristically determines if a pandas Series should be treated as real/continuous.
    A common heuristic is to check if the series is numeric and has a high
    number of unique values.
    """
    threshold = 10
    if not pd.api.types.is_numeric_dtype(y):
        return False
    return y.nunique() > threshold

def entropy(Y: pd.Series) -> float:
    """Calculates the entropy for a given series of labels."""
    class_counts = Y.value_counts()
    probs = class_counts / len(Y)
    # Epsilon prevents log2(0) for pure nodes where a probability is 0.
    epsilon = 1e-9
    return (-probs * np.log2(probs + epsilon)).sum()

def gini_index(Y: pd.Series) -> float:
    """
    Calculates the Gini impurity for a given series of labels.
    Gini impurity is a measure of how often a randomly chosen element
    from the set would be incorrectly labeled. A lower value means higher purity.
    """
    class_counts = Y.value_counts()
    probs = class_counts / len(Y)
    return 1 - (probs ** 2).sum()

def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    """
    Calculates the information gain for a given split.
    - For discrete (classification) outputs, it measures the reduction in impurity (Entropy/Gini).
    - For real (regression) outputs, it measures the reduction in variance (MSE).
    """
    if check_ifreal(Y):  # Regression Case
        parent_variance = Y.var()
        if parent_variance is None or pd.isna(parent_variance):
            parent_variance = 0
            
        weighted_child_variance = 0.0
        for value in attr.unique():
            child_Y = Y[attr == value]
            if len(child_Y) > 0:
                weight = len(child_Y) / len(Y)
                child_var = child_Y.var()
                if child_var is None or pd.isna(child_var):
                    child_var = 0
                weighted_child_variance += weight * child_var
        return parent_variance - weighted_child_variance
    else:  # Classification Case
        impurity_func = gini_index if criterion == 'gini_index' else entropy
        parent_impurity = impurity_func(Y)
        weighted_child_impurity = 0.0
        for value in attr.unique():
            child_Y = Y[attr == value]
            if len(child_Y) > 0:
                weight = len(child_Y) / len(Y)
                weighted_child_impurity += weight * impurity_func(child_Y)
        return parent_impurity - weighted_child_impurity

def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion: str, features: list) -> tuple:
    """
    Finds the best feature and value to split on by iterating through all
    features and their possible split points to maximize information gain.
    """
    max_info_gain = -1.0
    best_attr = None
    best_split_value = None

    for attr in features:
        # For real-valued features, test potential thresholds between unique sorted values.
        if check_ifreal(X[attr]):
            unique_values = sorted(X[attr].unique())
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2
                temp_split = X[attr] <= threshold
                curr_ig = information_gain(y, temp_split, criterion)
                if curr_ig > max_info_gain:
                    max_info_gain = curr_ig
                    best_attr = attr
                    best_split_value = threshold
        # For discrete features, test each unique category as a potential split.
        else:
            for value in X[attr].unique():
                temp_split = X[attr] == value
                curr_ig = information_gain(y, temp_split, criterion)
                if curr_ig > max_info_gain:
                    max_info_gain = curr_ig
                    best_attr = attr
                    best_split_value = value
                    
    return best_attr, best_split_value

def split_data(X: pd.DataFrame, y: pd.Series, attribute: str, value) -> tuple:
    """Splits the data into two branches based on the attribute and value."""
    if check_ifreal(X[attribute]):
        mask = X[attribute] <= value
    else:
        mask = X[attribute] == value
    
    # Use the mask to split both features (X) and labels (y)
    X_left, y_left = X[mask], y[mask]
    X_right, y_right = X[~mask], y[~mask]
    return X_left, y_left, X_right, y_right

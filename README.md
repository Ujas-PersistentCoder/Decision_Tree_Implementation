# Decision Tree from Scratch in Python

This repository contains a complete implementation of a Decision Tree algorithm for both classification and regression tasks, built from the ground up in Python. The model is implemented using only NumPy and pandas, without relying on high-level machine learning libraries like scikit-learn for the core tree logic.

The project demonstrates a deep understanding of the inner workings of decision trees, from the mathematical foundations of splitting criteria to the recursive logic of tree construction and model evaluation.

---

## ✨ Key Features

- **Handles Multiple Data Scenarios**: The tree is robustly designed to work with:
    - Real-valued features, discrete output (classification)
    - Real-valued features, real output (regression)
    - Discrete features, discrete output (classification)
    - Discrete features, real output (regression)
- **Standard Splitting Criteria**: Implements key criteria for choosing the best split:
    - **For Classification**: Gini Index and Entropy (Information Gain).
    - **For Regression**: Variance Reduction (Mean Squared Error).
- **Customizable Tree Depth**: Controls model complexity and prevents overfitting via a `max_depth` hyperparameter.
- **Tree Visualization**: Includes a `plot()` method to print a clear, text-based representation of the learned tree structure.

---

## 📂 Repository Structure

The repository is organized to separate the core algorithm from any example usage scripts.

- `tree/`: This directory is a Python package containing the core algorithm.
  - `__init__.py`: Makes the `tree` directory a package.
  - `utils.py`: Holds all the core mathematical and utility functions.
  - `base.py`: Contains the main `Node` and `DecisionTree` class definitions.
- `example.py`: A simple script demonstrating how to use the `DecisionTree` class.
- `decision_tree_complexity.md`: A detailed analysis of the algorithm's time complexity.

---

## 🚀 How to Use

To see the decision tree in action, please run the example script from your terminal:

```bash
python example.py

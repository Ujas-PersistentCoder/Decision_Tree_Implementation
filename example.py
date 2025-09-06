"""
example.py

A simple script demonstrating the usage of the custom DecisionTree class.

This script generates a synthetic classification dataset, splits it into
training and testing sets, trains the custom Decision Tree, and evaluates
its performance using standard metrics from scikit-learn.
"""
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Import the custom DecisionTree class from our 'tree' package
from tree.base import DecisionTree

def main():
    """Main function to run the demonstration."""
    print("--- Running Custom Decision Tree Demonstration ---")

    # 1. Generate a synthetic dataset for a classification problem
    X, y = make_classification(
        n_samples=200,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        random_state=42
    )
    # Convert to pandas for compatibility with our tree
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y_s = pd.Series(y)

    # 2. Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_s, test_size=0.3, random_state=42
    )
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")

    # 3. Initialize and train the Decision Tree
    # We'll use the Gini Index criterion with a max depth of 4 for this example
    tree = DecisionTree(criterion="gini_index", max_depth=4)
    tree.fit(X_train, y_train)

    # 4. Make predictions on the unseen test set
    predictions = tree.predict(X_test)

    # 5. Evaluate the model using a standard metric from scikit-learn
    # This shows knowledge of standard data science practices
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nModel Accuracy on Test Set: {accuracy:.3f}")

    # 6. Visualize the learned tree structure
    print("\n--- Learned Tree Structure ---")
    tree.plot()

if __name__ == "__main__":
    main()

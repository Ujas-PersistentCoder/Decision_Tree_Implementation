# Understanding the Time Complexity of a Decision Tree

This document provides an analysis of the time complexity for building (training) and using (predicting with) the custom-built decision tree algorithm. Understanding complexity is crucial for knowing how an algorithm will perform as the size of the data scales.

---

## 1. Training Time Complexity: `O(M * N log N)`

The process of building a decision tree is computationally intensive. The theoretical time complexity is approximately **O(M * N log N)**, where:
- **M** is the number of features.
- **N** is the number of samples (data points).

Let's break down where each component comes from:

### The Work Done at a Single Node: `O(M * N)`
To decide the best way to split a single node, the algorithm must perform an exhaustive search:
1.  It must loop through **every feature (M)** to see which one is the best candidate for a split.
2.  For **each feature**, it must evaluate every possible split point. This process involves scanning through all **N samples** at that node to calculate the information gain or variance reduction.

Therefore, the work to find the best split at one node is proportional to **M × N**.

### Repeating the Work for Every Node: `O(log N)`
The algorithm doesn't just split one node; it does this recursively to build the entire tree. For a reasonably balanced tree, the depth (the number of levels) is proportional to **log N**.

Since the `O(M * N)` work is performed at each level of the tree, the total complexity becomes the work per level multiplied by the number of levels.

**Total Training Time = (Work per Level) × (Number of Levels)**
`O(M * N) * O(log N) = O(M * N log N)`



---

## 2. Prediction Time Complexity: `O(depth)`

Predicting a result for a new data point is extremely fast compared to training. The theoretical time complexity for predicting a single sample is **O(depth)**, where `depth` is the maximum depth of the tree.

### The Prediction Path
To make a prediction, the algorithm traverses a single path from the root of the tree down to a leaf node. At each node, it performs a single comparison and moves to the next. The number of comparisons is equal to the depth of the tree.

Crucially, prediction time is **independent of the number of training samples (N) and features (M)**, as it only checks one feature at each level of the path.



For predicting an entire test set with **N_test** samples, the complexity is simply **O(N_test * depth)**.

---

## A Note on Practical Performance

While the theoretical complexity applies to all data scenarios, the practical runtime can differ significantly based on the type of input features.

### **Discrete Input Features (Fast Case)**
Finding the best split for a discrete (e.g., binary) feature is very fast as there are few split points to evaluate.

### **Real (Continuous) Input Features (Slow Case)**
This is the primary bottleneck in training. To find the best split for a single real-valued feature, the algorithm must test a split point between every pair of unique adjacent values. This intensive search is why the **"Real Input" cases are significantly slower** in practice than the "Discrete Input" cases.

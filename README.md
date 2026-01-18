# Optical Character Recognition (OCR) with Random Forest

## Overview
This project processes and recognizes handwritten digits using a **Random Forest** classifier implemented from scratch in C++. The system inputs 28x28 pixel grayscale images (representing digits 0-9) and predicts the correct digit. It is designed to work with standard datasets like MNIST.

## 📁 Project Structure

*   **`main.cpp`**
    *   **Role**: Entry point of the application.
    *   **Functionality**:
        *   Reads training and testing data from CSV files (`mnist_train.csv`, `mnist_test.csv`).
        *   Initializes the `RandomForest` with a specified number of trees.
        *   Trains the model using the training set.
        *   Evaluates the model against the test set and calculates precision.
*   **`randomForest.cpp` / `.h`**
    *   **Role**: Implementation of the ensemble learning algorithm.
    *   **Functionality**:
        *   Manages a collection of `Node` objects (decision trees).
        *   **Bagging**: Randomly samples the training data (80% of total size per tree) to train each tree independently, ensuring diversity.
        *   **Voting**: Aggregates predictions from all trees; the class with the most votes becomes the final prediction.
*   **`decisionTree.cpp` / `.h`**
    *   **Role**: Core logic for building individual decision trees.
    *   **Functionality**:
        *   implements the ID3/C4.5 generation algorithm.
        *   **Entropy & Information Gain**: Calculates Shannon Entropy to evaluate the purity of data splits.
        *   **Split Finding**: Finds the best feature (pixel) and value to split the data to maximize Information Gain.
        *   **Random Feature Selection**: Considers only a random subset of dimensions (`sqrt(total_dimensions)`) at each split to further decorrelate trees.

## 🧠 Data Representation
The images are flattened into 1D vectors for processing:
*   **Input**: 28x28 pixel images.
*   **Vector Size**: 785 integers.
    *   **Index 0**: The label (the actual digit, 0-9).
    *   **Indices 1-784**: The pixel intensity values (0-255).

## 🚀 Algorithm Details

### 1. Training (Recursive)
For each node in a Decision Tree:
1.  **Check Purity**: If all samples belong to the same class, create a **Leaf Node** with that class.
2.  **Find Best Split**:
    *   Select `sqrt(784)` random unique pixels (dimensions).
    *   For each dimension, test all unique values as potential split points.
    *   Calculate **Information Gain (IG)**:
        `IG = Entropy(Parent) - WeightedSum(Entropy(Children))`
    *   Choose the dimension and value that maximizes IG.
3.  **Split Data**: Divide samples into Left (<= cutoff) and Right (> cutoff) subsets.
4.  **Recurse**: Repeat the process for the left and right children.

### 2. Prediction
*   An unseen image vector is passed to the Random Forest.
*   It traverses every Decision Tree in the forest, ending at a leaf node that provides a vote.
*   The forest counts the votes for each digit (0-9).
*   The digit with the highest count is returned as the prediction.

## 🛠️ Build and Usage

### Prerequisites
*   G++ Compiler (supporting C++11 or higher)
*   Make
*   **Data Setup**:
    The training and testing data are compressed to save space. Before running the application, unzip the data archive:
    ```bash
    unzip mnist_data.zip
    ```

### Compilation
To build the main executable:
```bash
make
```
This produces an executable named `ex`.

### Execution
Run the application:
```bash
./ex
```
*Note: The program requires `mnist_train.csv` and `mnist_test.csv` to be in the same directory.*

### Testing
To run the unit test suite and verify the implementation:
```bash
./check.sh
```
This script validates individual components (entropy, split logic, etc.) and the overall model precision.

## 🧹 Cleaning
To remove compiled object files and executables:
```bash
make clean
```

---
**Copyright**: Mitroi Eduard Ionut

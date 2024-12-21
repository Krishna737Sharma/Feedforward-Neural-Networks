# Feedforward Neural Networks

## Objective
The objective of this assignment is to build a feedforward neural network to predict values using the Bike Sharing dataset. The assignment includes tasks such as preprocessing the dataset, designing the neural network model, optimizing it, performing cross-validation, and evaluating its performance.

---

## Tasks and Marks Distribution

### **Task 1: Load and Preprocess the Dataset** [1 Mark]
- Load the Bike Sharing dataset.
- Check for missing and duplicate values and remove them if found.

### **Task 2: Data Preprocessing** [1 Mark]
- Perform one-hot encoding of categorical features when necessary (not necessary if there is a natural ordering between categories).
- Perform feature scaling to normalize the data.

### **Task 3: Data Splitting** [1 Mark]
- Divide the data into training, validation, and test sets in the proportions of 70%, 15%, and 15%.

### **Task 4: Model Design** [7 Marks]
- Design a feedforward neural network with **N hidden layers** using the following specifications:
  - The number of units in each hidden layer decreases sequentially as 128, 64, 32, and so on.
  - Use **ReLU** activation for each hidden layer.
  - Choose an appropriate activation function for the output layer based on the prediction problem and implement it accordingly.

### **Task 5: Cost Function and Optimization** [3 Marks]
- Define a suitable cost function for this regression problem.
- Use the **SGD optimizer** to optimize the cost function via backpropagation.

### **Task 6: Cross-Validation** [2 Marks]
- Perform cross-validation using grid search to find the optimal value of **N** (number of hidden layers).

### **Task 7: Loss Plots** [1 Mark]
- Plot training and validation losses for each value of **N** on the same graph.

### **Task 8: Best Number of Hidden Layers** [2 Marks]
- Determine the best value of **N** obtained from cross-validation.
- Provide a justification for your choice.

### **Task 9: Test Set MSE** [1 Mark]
- Report the Mean Squared Error (MSE) on the test set using the model trained with the best value of **N**.

### **Task 10: Scatter Plot** [1 Mark]
- Create a scatter plot showing the predictions versus the true values for the best model obtained.

---

## Deliverables
- Python code implementing all the tasks.
- A report summarizing the findings, including:
  - Cross-validation results.
  - Loss plots for different values of **N**.
  - Scatter plot of predictions versus true values.
  - Justification for the selected number of hidden layers.
- README file (this document).

---

## Dataset Description
- **Dataset:** Bike Sharing dataset
- **Features:**
  - Contains categorical and numerical features relevant to bike sharing.
- **Target Variable:**
  - The target variable is continuous, representing the count of bikes shared.

---


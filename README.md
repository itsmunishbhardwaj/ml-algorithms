# ML-From-Scratch: An OOP Approach to Machine Learning

This repository contains a lightweight Machine Learning library built from the ground up using Object-Oriented Programming (OOP) and Low-Level Design (LLD) principles.

## 1. System Architecture (LLD)
The library follows a strict inheritance hierarchy to ensure code reusability and modularity.

- **BaseModel**: An abstract interface defining the `fit` and `predict` contract.
- **SimpleLinearModel**: A base class managing protected weights (`_weight`, `_bias`) and encapsulation.
- **LinearRegression**: Implementation of the Ordinary Least Squares (OLS) training algorithm.
- **LogisticRegressionModel**: Extension using the Sigmoid activation function for probabilistic classification.

---

## 2. Core Implementation

```python
import math

class BaseModel:
    """Abstract Base Class for all models."""
    def fit(self, X, y):
        raise NotImplementedError("Each model needs to have its own fit method")
    
    def predict(self, X):
        raise NotImplementedError("Each model needs to have its own predict logic")

class SimpleLinearModel(BaseModel):
    """Handles parameter storage and encapsulation."""
    def __init__(self, initial_weight, initial_bias):
        self._weight = initial_weight
        self._bias = initial_bias

    def update_weights(self, new_weights, new_bias):
        """Gatekeeper method for parameter updates."""
        if new_weights == 0 or new_bias == 0:
            print("Warning: Zero weight update detected!")
        self._weight = new_weights
        self._bias = new_bias
        return (self._weight, self._bias)
    
    def predict(self, x):
        return (self._weight * x) + self._bias

class LinearRegression(SimpleLinearModel):
    """Linear Regression using Ordinary Least Squares (OLS)."""
    def fit(self, X, y):
        n = len(X)
        mean_x = sum(X) / n
        mean_y = sum(y) / n

        numerator = 0
        denominator = 0

        for i in range(n):
            numerator += (X[i] - mean_x) * (y[i] - mean_y)
            denominator += (X[i] - mean_x) ** 2
        
        w = numerator / denominator
        b = mean_y - (mean_x * w)
        self.update_weights(w, b)

class LogisticRegressionModel(SimpleLinearModel):
    """Logistic Regression for binary classification."""
    def _sigmoid(self, z):
        return 1 / (1 + math.exp(-z))

    def predict(self, x):
        z = (self._weight * x) + self._bias
        return self._sigmoid(z)
    
    def predict_class(self, x):
        """Returns 0 or 1 based on a 0.5 probability threshold."""
        return 1 if self.predict(x) >= 0.5 else 0
```

## 3. Usage & Execution

### Training a Linear Model

```Python
X_train = [1, 2, 3, 4, 5]
y_train = [3, 5, 7, 9, 11] # y = 2x + 1

model = LinearRegression(0, 0)
model.fit(X_train, y_train)

print(f"Prediction for x=10: {model.predict(10)}") # Output: 21.0
```
### Classification Output
```Python
clf = LogisticRegressionModel(1.0, 0.0)
print(f"Class for x=5: {clf.predict_class(5)}") 
```
### Terminal Logs:

```Plaintext
Model trained. Weights: 2.0, Bias: 1.0
Prediction for x=10: 21.0
Class for x=5: 1
```

## 4. Key OOP Pillars Applied
1. **Abstraction**: Use of BaseModel to enforce a standard API.

1. **Encapsulation**: Use of _weight to protect internal state.

1. **Inheritance**: LinearRegression inherits prediction logic from SimpleLinearModel.

1. **Polymorphism**: Ability to iterate through multiple model types and call .predict() uniformly.


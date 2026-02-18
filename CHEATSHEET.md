# Python OOP for ML Engineers: From Zero to Hero
This cheatsheet tracks the evolution of a SimpleLinearModel, demonstrating the four pillars of Object-Oriented Programming.

## Phase 1: The Blueprint (Classes & Instances)
Goal: Define a template for a model and create an object in memory.

```Python
class SimpleLinearModel:
    # The Constructor: Defines what data the model holds
    def __init__(self, initial_weight, initial_bias):
        self.weight = initial_weight
        self.bias = initial_bias

# Instantiation: Creating the actual object
my_model = SimpleLinearModel(0.5, 0.1)

print(f"{my_model.weight=}") # Output: 0.5
```
## Phase 2: Data Protection (Encapsulation)
Goal: Protect internal parameters and use "gatekeeper" methods to update them.

```Python
class SimpleLinearModel:
    def __init__(self, initial_weight, initial_bias):
        # Use '_' prefix to signal "Private/Internal" variables
        self._weight = initial_weight
        self._bias = initial_bias

    def update_weights(self, new_weights, new_bias):
        # Validation Logic (The Gatekeeper)
        if new_weights == 0 or new_bias == 0:
            return "Warning: Zero weight update detected!"
        
        self._weight = new_weights
        self._bias = new_bias
        return (self._weight, self._bias)
    
    def predict(self, x):
        return (self._weight * x) + self._bias

my_model = SimpleLinearModel(0.5, 0.1)
print(my_model.update_weights(0, 0.5)) # Triggers the warning
```

## Phase 3: Building on Giants (Inheritance)
Goal: Reuse the foundation of an existing class to build a specialized version.

```Python
# LogisticRegressionModel "is a" SimpleLinearModel
class LogisticRegressionModel(SimpleLinearModel):
    def __init__(self, weight, bias):
        # Use super() to call the Parent's constructor
        super().__init__(weight, bias)

    # Method Overriding: Changing behavior for this specific child
    def predict(self, x):
        return "Add sigmoid calculations later"
```
## Phase 4: Flexible Interfaces (Polymorphism)
Goal: Treat different objects the same way if they share the same method names.

```Python
models = [
    SimpleLinearModel(0.5, 0.1), 
    LogisticRegressionModel(0.8, 0.2)
]

for model in models:
    # The loop doesn't care which model it is; it just calls .predict()
    print(f"{model.__class__.__name__} prediction: {model.predict(10)}")
```

### Key Takeaways for ML
- `Self`: Represents the specific instance (the specific set of weights).

- `init`: Where we define model architecture/hyperparameters.

- `Methods`: Actions like .fit(), .predict(), or .evaluate().

- `Private (_)`: Protects weights from being accidentally overwritten.
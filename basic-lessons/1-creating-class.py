#define class
class SimpleLinearModel:
    # define constructor
    def __init__(self, initial_weight, initial_bias):
        self._weight = initial_weight
        self._bias = initial_bias

    def update_weights(self, new_weights, new_bias):
        self._weight = new_weights
        self._bias = new_bias
        return(self._weight, self._bias)
    
    # define a method to predict y
    def predict(self, x):
        y = (self._weight * x ) + self._bias
        return(y)


my_model = SimpleLinearModel(0.5, 0.1)

#print(f"{my_model.weight=}")
print(f"{my_model.predict(10)=}")

# update weights
new_weights = my_model.update_weights(1, 0.5)
print(f"{new_weights}")
import math

# Set a base class from which everybody else takes the basics from
class BaseModel:
    def fit(self, X, y):
        raise NotImplementedError("Each model needs to have its own fit method")
    
    def predict(self, X):
        raise NotImplementedError("Each model needs to have its own predict logic")

#define class
class SimpleLinearModel(BaseModel):
    # define constructor
    def __init__(self, initial_weight, initial_bias):
        self._weight = initial_weight
        self._bias = initial_bias

    def update_weights(self, new_weights, new_bias):
        if new_weights == 0 or new_bias == 0:
            return(f"Warning: Zero weight update detected!")
        self._weight = new_weights
        self._bias = new_bias
        return(self._weight, self._bias)
    
    # define a method to predict y
    def predict(self, x):
        y = (self._weight * x ) + self._bias
        return(y)

class LinearRegression(SimpleLinearModel):
    def fit(self,X,y):
        n = len(X)
        mean_x = sum(X)/n
        mean_y = sum(y)/n

        # Calculate weight (w) and bias (b)
        numerator = 0
        denominator = 0

        for i in range(n):
            numerator += (X[i] - mean_x)*(y[i] - mean_y)
            denominator += (X[i] - mean_x)*(X[i] - mean_x)
        
        w = numerator/denominator
        b = mean_y - (mean_x * w)

        self.update_weights(w,b)

class LogisticRegressionModel(SimpleLinearModel):
    def __init__(self,weight, bias):
        super().__init__(weight, bias)

    def predict(self, x):
        # Liner value z = wx + b
        z = (self._weight * x) + self._bias
        return(self._sigmoid(z))
    
    def _sigmoid(self, z):
        z_sigmoid = 1 / (1 + math.exp(-z))
        return (z_sigmoid)
    
    def predict_class(self, x):
        result = self.predict(x)
        if result >= 0.5:
            return(1)
        else:
            return(0)
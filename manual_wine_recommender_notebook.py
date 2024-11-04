# Example neural network: pick red or white wine based on weather and food
# Inputs: weather (warm = 0, cold = 1) and food pairing (seafood = 0, vegetables = 0.5, meat = 1)]
# Outputs: red wine = 1, white wine = 0

"""
Inputs		Target
[0,0]		0 
[0,0.5]		0
[0,1]		1
[1,0]		0
[1,0.5]		1
[1,1]		1
"""

# Defining example input vector and weights (for vegetables on a sunny day)
input_vector = [0, 0.5] # warm weather, vegetables
weights = [1,3] # food is 3x more important than weather

# Computing the dot product of example input_vector and weights manually
first_indexes_mult = input_vector[0] * weights[0]
second_indexes_mult = input_vector[1] * weights[1]
dot_product = first_indexes_mult + second_indexes_mult
print(f"The dot product is: {dot_product}")
#The dot product is: 1.5

# Computing the dot product with numpy
import numpy as np
dot_product = np.dot(input_vector, weights)
print(f"The dot product is: {dot_product}")
#The dot product is: 1.5

# Wrapping the vectors in numpy arrays
input_vector = np.array([0, 0.5])
weights = np.array([1,3])
bias = np.array([-1.25])

# Defining the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Defining the layers
def make_prediction(input_vector, weights, bias):
    layer_1 = np.dot(input_vector, weights) + bias
    layer_2 = sigmoid(layer_1)
    return layer_2

# Make the prediction
prediction = make_prediction(input_vector, weights, bias)
print(f"The prediction result is: {prediction}")
# The prediction result is: [0.5621765]; as this is > 0.5, this would predict 1 (= red wine), which is incorrect

# Compute the error
target = 0
mse = np.square(prediction - target)
print(f"Prediction: {prediction}; Error: {mse}")
# Prediction: [0.5621765]; Error: [0.31604242]

# Updating the weights
def sigmoid_deriv(x):
   return sigmoid(x) * (1-sigmoid(x))
derror_dprediction = 2 * (prediction - target)
layer_1 = np.dot(input_vector, weights) + bias
dprediction_dlayer1 = sigmoid_deriv(layer_1)
dlayer1_dweights = 0*weights + 1*input_vector
derror_dweights = (
    derror_dprediction * dprediction_dlayer1 * dlayer1_dweights
)
weights = weights - derror_dweights
print(f"Updated weights: {weights}")
# Updated weights: [1, 2.8616292]

# Updating the bias
dlayer1_dbias = 1
derror_dbias = (
   derror_dprediction * dprediction_dlayer1 * dlayer1_dbias
)
bias = bias - derror_dbias
print(f"Updated bias: {bias}")
# Updated bias: [-1.52674159]

# Updating the prediction
prediction = make_prediction(input_vector, weights, bias)
print(f"The prediction result is: {prediction}")
# The prediction result is: [0.47603662]; as this is < 0.5, this would predict 0 (= white wine), which is correct

# Test a couple of different input vectors
input_vector = [0, 0] # warm weather, seafood
prediction = make_prediction(input_vector, weights, bias)
print(f"The prediction result is: {prediction}")
# The prediction result is: [0.17847093]; as this is < 0.5, this would predict 0 (= white wine), which is correct
input_vector = [1, 0.5] # cold weather, vegetables
prediction = make_prediction(input_vector, weights, bias)
print(f"The prediction result is: {prediction}")
# The prediction result is: [0.71178579]; as this is > 0.5, this would predict 1 (= red wine), which is correct
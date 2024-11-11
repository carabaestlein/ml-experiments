# Import NN Class and pyplot
import numpy as np
import matplotlib.pyplot as plt
from two_layer_NN import NeuralNetwork

# Define raw input vectors and output targets data
input_vectors_raw = [
    ['warm', 'seafood'],
    ['warm', 'vegetables'],
    ['warm', 'meat'],
    ['cold', 'seafood'],
    ['cold', 'vegetables'],
    ['cold', 'meat']
]
targets_raw = ['white', 'white', 'red', 'white', 'red', 'red']

# Map raw data to encoded data
input_mapping = {'warm': 0, 'cold': 1, 'seafood': 0, 'vegetables': 0.5, 'meat': 1}
targets_mapping = {'white': 0, 'red': 1}
input_vectors = np.array([[input_mapping[item[0]], input_mapping[item[1]]] for item in input_vectors_raw])
targets = np.array([targets_mapping[color] for color in targets_raw])

# Verify encoded input vectors and output targets
print(f"Input vectors: {input_vectors}")
# Input vectors: [[0.  0. ] [0.  0.5] [0.  1. ] [1.  0. ] [1.  0.5] [1.  1. ]]
print(f"Output targets: {targets}")
# Output targets: [0 0 1 0 1 1]

# Initialise and train NN
learning_rate = 0.1
neural_network = NeuralNetwork(learning_rate)
training_error = neural_network.train(input_vectors, targets, 10000)

# Plot training error
plt.plot(training_error)
plt.xlabel("Iterations")
plt.ylabel("Error for all training instances")
plt.savefig("simple_wine_recommender_NN_cumulative_error.png")

# Print model parameters
print(f"Model weights: {neural_network.weights}")
# Model weights: [3.61029011 7.89077727]
print(f"Model bias: {neural_network.bias}")
# Model bias: -5.695445789768399

# Make a couple of predictions
input_vector = [0, 0.5] # warm weather, vegetables
prediction = neural_network.predict(input_vector)
print(f"Prediction: {prediction}") # this is < 0.5, therefore = white wine, which is correct
input_vector = [1, 1] # cold weather, meat
prediction = neural_network.predict(input_vector) # this is > 0.5, therefore = red wine, which is correct
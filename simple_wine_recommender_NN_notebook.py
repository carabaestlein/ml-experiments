# Import NN Class and pyplot
import numpy as np
import matplotlib.pyplot as plt
from two_layer_NN import NeuralNetwork

# Define input vectors, targets and learning rate
input_vectors = np.array(
   [
       [0, 0],
       [0, 0.5],
       [0, 1],
       [1, 0],
       [1, 0.5],
       [1, 1],
   ]
)
targets = np.array([0, 0, 1, 0, 1, 1])
learning_rate = 0.1

# Initialise and train NN
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

# Make a prediction
input_vector = [0, 0.5] # warm weather, vegetables
prediction = neural_network.predict(input_vector)
print(f"Prediction: {prediction}") # this is < 0.5, therefore = white wine, which is correct
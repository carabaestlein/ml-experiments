import numpy as np
from two_layer_NN import NeuralNetwork

# Map raw data to encoded data
input_mapping = {
    'warm': 0,
    'cold': 1,
    'seafood': 0,
    'vegetables': 0.5,
    'meat': 1
}
targets_mapping = {'white': 0, 'red': 1}

def encode_input(input_vector_raw):
    """Encode raw input vector into numerical values."""
    return np.array([input_mapping[item] for item in input_vector_raw])

def encode_target(target_raw):
    """Encode the target label into a numerical value."""
    return targets_mapping[target_raw]

def train_neural_network():
    """Train the neural network with predefined data and return the trained model."""
    input_vectors_raw = [
        ['warm', 'seafood'],
        ['warm', 'vegetables'],
        ['warm', 'meat'],
        ['cold', 'seafood'],
        ['cold', 'vegetables'],
        ['cold', 'meat']
    ]
    targets_raw = ['white', 'white', 'red', 'white', 'red', 'red']

    # Encode the input and target data
    input_vectors = np.array([encode_input(item) for item in input_vectors_raw])
    targets = np.array([encode_target(color) for color in targets_raw])

    # Initialize and train the neural network
    learning_rate = 0.1
    neural_network = NeuralNetwork(learning_rate)
    neural_network.train(input_vectors, targets, 10000)

    return neural_network

def make_prediction(input_vector_raw, neural_network):
    """Make a prediction based on the raw input vector."""
    # Encode the raw input vector
    encoded_input = encode_input(input_vector_raw)

    # Get the neural network prediction
    prediction = neural_network.predict(encoded_input)

    # Convert prediction to raw output ('white' or 'red')
    predicted_color = 'red' if prediction >= 0.5 else 'white'
    return predicted_color

def main():
    # Train the neural network
    neural_network = train_neural_network()

    # Accept raw input and make prediction
    raw_input = input("Select a combination of weather (warm, cold) and food (seafood, vegetables, meat), i.e. ['cold', 'meat']")
    raw_input = eval(raw_input)  # Convert input string to list

    # Ensure the input is valid
    if len(raw_input) != 2 or not all(isinstance(x, str) for x in raw_input):
        print("Invalid input. Please select two valid options.")
        return

    # Make prediction and display result
    prediction = make_prediction(raw_input, neural_network)
    print(f"The recommended wine for {raw_input} is: {prediction}")

if __name__ == "__main__":
    main()

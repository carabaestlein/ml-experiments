# Cara's ML experiments

Learning how neural networks work by building a simple manual one to recommend wine (white or red) based on the weather (warm or cold) and accompanying food (seafood, vegetables or meat).

`learning_the_concepts_notebook.py` walks through computing dot products, creating the different layers of a NN, making predictions, computing errors and then adjusting the model parameters via backpropagation.

`two_layer_NN.py` contains the code for a simple two layer NN, including the code to train it over many iterations. Note that given the training data is small this model will be heavily overfitted.

`simple_wine_recommender_NN_notebook.py` contains the code to train the two layer NN on the following data set:

| Input              | Encoded input | Target output | Encoded target output |
| ------------------ | ------------- | ------------  | --------------------- |
| [warm, seafood]    | [0, 0]        | [white]       | [0]                   |
| [warm, vegetables] | [0, 0.5]      | [white]       | [0]                   |
| [warm, meat]       | [0, 1]        | [red]         | [1]                   |
| [cold, seafood]    | [1, 0]        | [white]       | [0]                   |
| [cold, vegetables] | [1, 0.5]      | [red]         | [1]                   |
| [cold, meat]       | [1, 1]        | [red]         | [1]                   |

`simple_wine_recommender.py` accepts an input vector, such as ['cold', 'meat'], and outputs a wine recommendation 'wine' or 'red'.
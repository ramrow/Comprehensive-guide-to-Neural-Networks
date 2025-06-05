# Comprehensive Guide to Neural Networks
Summer 2025 built for a more personalized experience working with neural networks, helps build understanding surrounding the math and complexity behind the model. Please keep in mind that this project is intended to give beginners a rundown of the important aspects of a neural network, and will not cover variations like liquid networks, Kolmogorov-Arnold networks, or etc, nor will this guide cover advance practices either.A simple model is built and tested on a set of pixelized numbers to identify between 1s and 5s
## Features
The data folder includes both the testing and training data after feature transformation was performed:
- Average intensity of for each grid where 5s will have higher intensity while 1s while have lower intensity. 
- Symmetry between the top half of the number and the bottom half of the number.
# Introduction to Neural Networks
Neural networks are powerful machine learning models that anyone can use and build. It is inspired by the human brain and can be used to understanding patterns through processing data. They are typically used in fields such as image or natural language processing.
## Structure
The most basic component of a neural network is called a node(inspired cells in the human brain) and it pocesses the ability to recieve and pass along information.

There are three main sections to a neural network: input layer, where data gets offloaded onto the model, the hidden layer, where calculations take place in order to produce a result, and finally an output layer, where the results fromr the model are spat out.

Another fundemental aspect of neural networks are the weights and biases: 
- Weights:
    - Each weight corresponds to an input and is mutiplied to its corresponding input during forward propagation.
    - The weights represent how much influence an input has on the output by scaling it through mutiplication.
    - The value of the weights are initially set to an array of zeros in this repo's neural network but can be anything that is minimalistic, will not impact the output in any way.
    - The training of the neural network basically means the adjustments of the weights to better fit the training data through forward and backward propagation. This process refines the model and allows the model to make predictions on tests.
- Biases:
    - Each bias adds a layer of flexibility to the model it is applied to the summation of all weights mutiplied to each input(node) in each layer.
    - It is used to expand the range of the threshold in which a node can be activated through the activation function.
    - Flexibility is extremely important as sharp thresholds aren't likely to exist in a real world setting. The training of the neural network also updates the biases for each iteration in order to finetune the model.
$$\displaystyle{  y =  \sum_{i}^n w_{i}x_{i} + b }\\ \textbf{ y is weighted sum of}\\ \textbf{ w is the list of weights}\\ \textbf{ x is the list of inputs}\\ \textbf{ b is the bias} $$



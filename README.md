# Comprehensive Guide to Neural Networks
Summer 2025 built for a more personalized experience working with neural networks, helps build understanding surrounding the math and complexity behind the model. Please keep in mind that this project is intended to: first and foremost, strengthen my own understanding of neural network, and secondly, give beginners a rundown of the important aspects of a neural network, and will not cover variations like liquid networks, Kolmogorov-Arnold networks, or etc, nor will this guide cover advance practices either.A simple model is built and tested on a set of pixelized numbers to identify between 1s and 5s. Please keep in mind that there can be errors in this guide and any feedback would be appreciated.
## Features
The data folder includes both the testing and training data after feature transformation was performed:
- Average intensity of for each grid where 5s will have higher intensity while 1s while have lower intensity. 
- Symmetry between the top half of the number and the bottom half of the number.
# Introduction to Neural Networks
Neural networks are powerful machine learning models that anyone can use and build. It is inspired by the human brain and can be used to understanding patterns through processing data. They are typically used in fields such as image or natural language processing.
## Structure
The most basic component of a neural network is called a node(inspired cells in the human brain) and it pocesses the ability to recieve and pass along information.

There are three main sections to a neural network: input layer, where data gets offloaded onto the model, the hidden layer, where calculations take place in order to produce a result, and finally an output layer, where the results fromr the model are spat out.

### Components 
- Weights:
    - Each weight corresponds to an input and is mutiplied to its corresponding input during forward propagation.
    - The weights represent how much influence an input has on the output by scaling it through mutiplication.
    - The value of the weights are initially set to an array of zeros in this repo's neural network but can be anything that is minimalistic, will not impact the output in any way.
    - The training of the neural network basically means the adjustments of the weights to better fit the training data through forward and backward propagation. This process refines the model and allows the model to make predictions on tests.
- Biases:
    - Each bias adds a layer of flexibility to the model it is applied to the summation of all weights mutiplied to each input(node) in each layer.
    - It is used to expand the range of the threshold in which a node can be activated through the activation function.
    - Flexibility is extremely important as sharp thresholds aren't likely to exist in a real world setting. The training of the neural network also updates the biases for each iteration in order to finetune the model.

$$ \displaystyle{ y =  \sum_{i}^n w_{i}x_{i} + b }$$
$$ \textbf{ y is weighted sum of}$$
$$ \textbf{ w is the list of weights}$$
$$ \textbf{ x is the list of inputs}$$
$$ \textbf{ b is the bias} $$

## Activation Function
Activation functions play an important role in determining whether or not a node should be activated or not. The activation function is applied to the output of a node in both the hidden and output layers. The functions are important because they introduce non-linearity to the model(non-linearity means that the relationship between input and output aren't a straight line), helps capture real world data where there can be overlaps(problems that are not linearly separable). Activation functions also provide a vital role in backpropagation by providing the gradient needed to update both the weights and biases.

Here are some examples of popular activation functions:
1. Sigmoid
    - Shaped like an S going from 0 to 1 concaving upwards from all x less than 0 and concaving downwards from all x greater than or equal to 0, useful for binary classifications as output ranges from 0 to 1
    - Defined as:
      
      $$A = \frac{1}{1+e^{-x}}$$
2. Tanh
    - Hyperbolic tangent function, basically just a variation of the sigmoid function, output ranges from -1 to 1
    - Defined as:
      
      $$tanh(x) = 2*sigmoid(2x)-1 = \frac{2}{1+e^{-2x}}-1$$
3. ReLU
    - Rectified Linear Unit, and only passes positive inputs as it is and will pass 0 if the input is less than or equal to 0. Less computationally expensive when compared to sigmoid and tanh due to simplier mathematical operations
    - Defined as:
      
      $$A(x) = max(0,x)$$

## Backpropagation
This is a the process that goes back from the output layer to the input layer and updates the weights and biases for the nodes. Backpropagation is the keystone of the neural network as it is what allows the network to "learn" and improve itself. The core of backpropagation is the chain rule from Calculus which allows the network to calculate how each weight contributes to error when comparing the output and the label, by calculating and mutiplying the graidents of the loss function and the hidden layers:

$$\frac{\partial loss}{\partial weight} = \frac{\partial loss}{\partial active} * \frac{\partial active}{\partial y} * \frac{\partial y}{\partial weight}$$

This gradient is then used to caliberate the weights of the network using the learning rate, an input parameter to control the size of the step taken towards the negative gradient

$$w_{new} = w_{old} - \eta*\frac{\partial loss}{\partial weight}$$

This process is repeated for a inputted number of times(# of epoches) in the model on this github repo in order to improve the accuracy of the model.
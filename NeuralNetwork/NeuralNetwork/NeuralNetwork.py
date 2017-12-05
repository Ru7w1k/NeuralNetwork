from numpy import exp, array, random, dot, append, insert, ndarray

class NeuronLayer():
    def __init__(self, no_of_neurons, no_of_inputs_per_neuron):
        self.no_of_neurons = no_of_neurons
        self.no_of_inputs_per_neuron = no_of_inputs_per_neuron
        self.weights = random.random((no_of_inputs_per_neuron, no_of_neurons))

class NeuralNetwork():
    def __init__(self, layers):
        self.layers = layers

    def ReLU(self, x):
        return x * (x > 0)

    def dReLU(self, x):
        return 1. * (x > 0)

    def train(self, training_input_set, training_output_set, epochs):
        for iteration in range(epochs):
            # outputs from all layers in Neural Network
            outputs = self.think(training_input_set)

            # store errors generated in each layer using BackProp
            # errors = ndarray(len(outputs))
            errors = []

            # store delta values for each layer
            # delta = ndarray(len(outputs))
            delta = []

            # store adjustments for each layer
            # adjustments = ndarray(len(outputs))
            adjustments = []

            # calculate Error and Delta for last layer (OUTPUT layer)
            last = len(outputs) - 1
            errors.append(training_output_set - outputs[last])
            delta.append(errors[0] * self.dReLU(outputs[last]))

            # repeat the same logic for other layers
            for i in range(last, 0, -1):
                errors.append(delta[last - i].dot(self.layers[i].weights.T))
                delta.append(errors[last - i] * self.dReLU(outputs[i-1]))

            # calculate Adjustment for layer one
            adjustments.append(training_input_set.T.dot(delta[0]))
            
            # repeat for the other layers
            for i in range(1, len(outputs)):
                adjustments.append(outputs[i-1].T.dot(delta[i]))
                
            # adjust the weights 
            for i in range(len(outputs)):
                self.layers[i].weights += adjustments[i]

    def think(self, inputs):
        output = []

        # calculate first output seprately
        # INPUT nodes -> 1st hideen layer
        output.append(self.ReLU(dot(inputs, self.layers[0].weights)))

        # repeat same logic for other layers
        # 1st hidden layer -> 2nd hidden layer .. 2 -> 3 ..  -> OUTPUT nodes
        for i in range(1, len(self.layers)):
            output.append(self.ReLU(dot(output[i-1], self.layers[i].weights)))

        # will contain the output from each layer,
        # last entry will be the final output on OUTPUT nodes layer
        return output

    def print_weights(self):
        for layer in self.layers:
            print(layer.weights)

if __name__ == '__main__':
    random.seed(1)

    layer1 = NeuronLayer(4, 3)

    # Create layer 2 (a single neuron with 4 inputs)
    layer2 = NeuronLayer(1, 4)

    # Combine the layers to create a neural network
    neural_network = NeuralNetwork([layer1, layer2])

    print("Stage 1) Random starting synaptic weights: ")
    neural_network.print_weights()

    # The training set. We have 7 examples, each consisting of 3 input values
    # and 1 output value.
    training_set_inputs = array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
    training_set_outputs = array([[0, 1, 1, 1, 1, 0, 0]]).T

    # Train the neural network using the training set.
    # Do it 60,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 60000)

    print ("Stage 2) New synaptic weights after training: ")
    neural_network.print_weights()

    # Test the neural network with a new situation.
    print ("Stage 3) Considering a new situation [1, 1, 0] -> ?: ")
    outputs = neural_network.think(array([1, 1, 0]))
    print (outputs[outputs.size - 1])



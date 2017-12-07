import numpy as np

# represents a neural layer containing neurons and their input weights 
class NeuralLayer():
    def __init__(self, no_of_neurons, no_of_inputs_to_neuron):
        self.noOfNeurons = no_of_neurons
        self.noOfInputsToNeuron = no_of_inputs_to_neuron
        self.weights = np.random.random((no_of_inputs_to_neuron , no_of_neurons)) 


# represents a neural network with multiple hidden layers
class NeuralNetwork():

    # inputSize: number of input neurons
    # hiddenSizes: array containing number of neurons in each hidden layer
    # outputSize: number of output neurons
    def __init__(self, inputSize, hiddenSizes, outputSize):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.hiddenLayers = []

        # generate hidden layers
        # first hidden layer will take input from INPUT neurons
        self.hiddenLayers.append(NeuralLayer(hiddenSizes[0], inputSize))
        
        # intermediate hidden layers will take input from previous hidden layer
        for i in range(1, len(hiddenSizes)):
            self.hiddenLayers.append(NeuralLayer(hiddenSizes[i], hiddenSizes[i - 1]))

        # last layer is OUTPUT layer
        self.hiddenLayers.append(NeuralLayer(outputSize, hiddenSizes[len(hiddenSizes) - 1]))


    # train the Neural Network on given inputs and labels
    # input: input data ( inputSize * n array containing n input values)
    # label: output data ( outputSize * n array containing n output labels)
    # epoch: number of epochs ( one fwd + one bwd on all input sets)
    def train(self, input, label, epoch):
        # train for epoch number of times
        for i in range(epoch):
            layersOutput = self.fwd_prop(input)
            cost =  np.sum(np.power(np.asarray(label) - np.asarray(layersOutput[len(layersOutput) - 1]), 2))
            layersDelta = self.back_prop(layersOutput, np.asarray(label).reshape((len(input),self.outputSize)))
            self.adjust_weights(layersDelta, (layersOutput), np.asarray(input))
            print('cost: ' + str(cost))

    
    # returns output generated by each layer from given input cases
    def fwd_prop(self, input):
        layersOutput = []

        # output generated by first hidden layer depends on INPUT layer
        # change input to adjust bias node
        #input = np.asarray(input)
        #bias = np.ones((input.shape[0],1))
        #input = np.concatenate((input, bias), axis=1)
        layersOutput.append(self.Sigmoid(np.dot(input, self.hiddenLayers[0].weights)))

        # output generated by intermediate hidden layers will depend on output of prev hidden layer
        for i in range(1, len(self.hiddenLayers)):
            #bias = np.ones((layersOutput[i-1].shape[0],1))
            #tmp = np.concatenate((layersOutput[i-1], bias), axis=1)
            layersOutput.append(self.Sigmoid(np.dot(layersOutput[i-1], self.hiddenLayers[i].weights)))

        return layersOutput

    # calculates error in each layer and adjusts the weight according to it
    def back_prop(self, layersOutput, expectedOutput):
        layersDelta = []
        last = len(self.hiddenLayers) - 1

        # calculate error and delta for output layer
        layersError = (expectedOutput - layersOutput[last])
        layersDelta.append(layersError * self.dSigmoid(layersOutput[last]))

        # calculate error and delta for other layers
        for i in range(last - 1, -1, -1):
            layersError = (np.dot(layersDelta[last - 1 - i], self.hiddenLayers[i + 1].weights.T))
            layersDelta.append(layersError * self.dSigmoid(layersOutput[i]))

        layersDelta.reverse()
        return layersDelta

    # adjust weights from layersDelta and inputs
    def adjust_weights(self, layersDelta, layersOutput, input):
        layersAdjustment = []
        
        # calculate adjustment for first layer 
        layersAdjustment.append(input.T.dot(layersDelta[0]))

        # calculate the adjustment for other layers
        for i in range(1, len(layersDelta)):
            layersAdjustment.append(layersOutput[i-1].T.dot(layersDelta[i]))
        
        # recalculate all the weights
        for i in range(len(layersAdjustment)):
            self.hiddenLayers[i].weights += layersAdjustment[i]
        
    # Activation Functions
    # Sigmoid Function
    def Sigmoid(self, x):
        #return x * (x > 0)
        return 1 / (1 + np.exp(-x))
        #return np.tanh(x)

    # Sigmoid Prime Function
    def dSigmoid(self, x):
        #return 1. * (x > 0)
        return x * (1 - x)
        #return 1.0 - np.power(np.tanh(x), 2)

    # End Activation Functions

    # print the structure of Neural Network
    def print_structure(self):
        print('Neural Network: ', end = '\n')
        print(' (' + str(self.inputSize) + ' ' + str(0) + ') ')
        for i in self.hiddenLayers:
            print(' (' + str(i.noOfNeurons) + ' ' + str(i.noOfInputsToNeuron) + ') : \n' + str(i.weights))


if __name__ == '__main__':
    NN = NeuralNetwork( 4, [10 for _ in range(10)], 4)
    NN.print_structure()
    #NN.train([[1.0,1.0], [1.0,0.0], [0.0,1.0]], [1.0, 0.0, 0.0], 100)
    input = [[0,0,0,0],[0,0,0,1],[0,0,1,0],[0,0,1,1],[0,1,0,0],[0,1,0,1],[0,1,1,0],[0,1,1,1],[1,0,0,0],[1,0,0,1],[1,0,1,0],[1,0,1,1]]
    label = [[0,0,0,0],[0,1,0,1],[0,0,0,0],[0,1,0,1],[0,0,0,0],[0,1,0,1],[0,0,0,0],[0,1,0,1],[1,0,1,0],[1,1,1,1],[1,0,1,0],[1,1,1,1]]
    #input = [[0,0],[0,1],[1,0], [1,1]]
    #input = [[2,4],[1,1],[3,8],[5,25],[4,16],[6,37],[8,65],[7,49],[9,81],[10,105]]
    #label = [[0,0], [0,0], [1,1]]
    #label = [0, 1,1,1]
    #label = [1, 1, 0, 1, 1, 0, 0, 1, 1, 0]
    NN.train(input, label, 50000)
    NN.print_structure()
    #output = NN.fwd_prop([1.0,1.0])
    output = NN.fwd_prop([11,121])
    #print('output: ' + str(output[len(output) - 1]))
    print('output: ' + str(output))
    output = NN.fwd_prop([4,16])
    print('output: ' + str(output))
    output = NN.fwd_prop([3,9])
    print('output: ' + str(output))
    output = NN.fwd_prop([6,36])
    print('output: ' + str(output))




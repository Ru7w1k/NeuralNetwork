import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

masterInput = [[0,0,0,0],[0,0,0,1],[0,0,1,0],[0,0,1,1],[0,1,0,0],[0,1,0,1],[0,1,1,0],[0,1,1,1],[1,0,0,0],[1,0,0,1],[1,0,1,0],[1,0,1,1],[1,1,0,0],[1,1,0,1],[1,1,1,0],[1,1,1,1]]
masterLabel = [[0,0,0,0],[0,1,0,1],[0,0,0,0],[0,1,0,1],[0,0,0,0],[0,1,0,1],[0,0,0,0],[0,1,0,1],[1,0,1,0],[1,1,1,1],[1,0,1,0],[1,1,1,1],[1,0,1,0],[1,1,1,1],[1,0,1,0],[1,1,1,1]]

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
    def __init__(self, inputSize, hiddenSizes, outputSize, lr=1):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.hiddenLayers = []
        self.lr = lr

        # generate hidden layers
        # first hidden layer will take input from INPUT neurons
        self.hiddenLayers.append(NeuralLayer(hiddenSizes[0], inputSize))
        
        # intermediate hidden layers will take input from previous hidden layer
        for i in range(1, len(hiddenSizes)):
            self.hiddenLayers.append(NeuralLayer(hiddenSizes[i], hiddenSizes[i - 1]))

        # last layer is OUTPUT layer
        self.hiddenLayers.append(NeuralLayer(outputSize, hiddenSizes[-1]))


    # train the Neural Network on given inputs and labels
    # input: input data ( inputSize * n array containing n input values)
    # label: output data ( outputSize * n array containing n output labels)
    # epoch: number of epochs ( one fwd + one bwd on all input sets)
    def train(self, input, label, epoch):
        # train for epoch number of times
        for i in range(epoch):
            layersOutput = self.fwd_prop(input)
            cost =  np.sum(np.power(np.asarray(label) - layersOutput[-1], 2))
            layersDelta = self.back_prop(layersOutput, np.asarray(label).reshape((len(input),self.outputSize)))
            self.adjust_weights(layersDelta, (layersOutput), np.asarray(input))
            print('cost: ' + str(cost) + ' ' + str((i/epoch)*100) + '%')
        
        #dump the NN to file
        pkl_out = open("nn2.pkl", "wb")
        pkl.dump(NN, pkl_out)
        pkl_out.close()
    

    
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
            self.hiddenLayers[i].weights += (self.lr * layersAdjustment[i])
        
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
    NN = NeuralNetwork( 4, [5, 4], 4, 0.3)
    NN.print_structure()
    #NN.train([[1.0,1.0], [1.0,0.0], [0.0,1.0]], [1.0, 0.0, 0.0], 100)
    input = [masterInput[i] for i in [0,1,4,5,6,10,11,12,14]]
    label = [masterLabel[i] for i in [0,1,4,5,6,10,11,12,14]]
    #input = [[0,0], [0,1], [1,0], [1,1]]
    #input = [[0,0], [0,1], [1,0], [1,1]]
    #input = [[2,4],[1,1],[3,8],[5,25],[4,16],[6,37],[8,65],[7,49],[9,81],[10,105]]
    #label = [[0,0], [0,0], [1,1]]
    #label = [[0], [1], [1], [1]]
    #label = [[0], [1], [1], [0]]
    #label = [1, 1, 0, 1, 1, 0, 0, 1, 1, 0]
    
    try:
        pkl_in = open("nn2.pkl", "rb")
        NN = pkl.load(pkl_in)
    except:
        NN.train(input, label, 50000)
    
    NN.train(input, label, 50000)
    NN.print_structure()

    outputCost = 0

    for i in range(0,16):
        output = NN.fwd_prop(masterInput[i])
        print('input : ' + str(masterInput[i]))
        print('output: ' + str(output[-1]))
        outputCost += np.sum(np.power(np.asarray(masterLabel[i]) - np.asarray(output[-1]), 2))

    #output = NN.fwd_prop([1,1,0,0])
    #print('output: ' + str(output))
    #outputCost += np.sum(np.power(np.asarray([1,0,1,0]) - np.asarray(output[-1]), 2))

    #output = NN.fwd_prop([1,1,0,1])
    #print('output: ' + str(output))
    #outputCost += np.sum(np.power(np.asarray([1,1,1,1]) - np.asarray(output[-1]), 2))

    #output = NN.fwd_prop([1,1,1,0])
    #print('output: ' + str(output))
    #outputCost += np.sum(np.power(np.asarray([1,0,1,0]) - np.asarray(output[-1]), 2))

    #output = NN.fwd_prop([1,1,1,1])
    #print('output: ' + str(output))
    #outputCost += np.sum(np.power(np.asarray([1,1,1,1]) - np.asarray(output[-1]), 2))

    print('test cost: ' + str(outputCost))

    #output = NN.fwd_prop([1,0,0,1])
    #print('output: ' + str(output))
    #output = NN.fwd_prop([0,0,1,0])
    #print('output: ' + str(output))
    #output = NN.fwd_prop([0,1,0,1])
    #print('output: ' + str(output))
    #output = NN.fwd_prop([1,1,1,1])
    #print('output: ' + str(output))




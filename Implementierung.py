#Autor: Boris Giba
#letzter Zugriff: 08.04.2019
#verwendet in: Facharbeit: "Modellierung, Implementierung und Strukturvergleich eigener neuronaler Netze zur Handschrifterkennung und Vergleich mit moderner Bibliothek" (2019)

import numpy as np
import matplotlib.pyplot as plt

from loadMNISTpkl import loadMNIST
from pathFinder import *
from DrawImage import *


def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y


class Dataset(object):
    """
    x_train: training data: array of arrays of float
    y_train: training labels: array of arrays of float
    x_test: test data: array of arrays of float
    y_test: test labels: array of arrays of float
    """
    def __init__(self,dataset,x_train=None,x_test=None,y_train=None,y_test=None):
        self.dataset=dataset
        self.x_train=x_train
        self.x_test=x_test
        self.y_train=y_train
        self.y_test=y_test


    def getImage(self,setName,index):
        if setName=="train":
            image=self.x_train[index]
        else:
            image=self.x_test[index]
            
        return image


    def showImage(self,setName,index):
        print(self.y_train[index])
        
        if setName=="train":
            label = self.y_train[index].argmax(axis=0)
            image = self.x_train[index].reshape([28,28])
        elif setName=="test":
            label = self.y_test[index].argmax(axis=0)
            image = self.x_test[index].reshape([28,28])
            
        plt.title('Example: %d  Label: %d' % (index, label))
        plt.imshow(image, cmap=plt.get_cmap("gray_r"))
        plt.show()


class Neuron(object):
    """
    value: float from interval [0;1]
    bias: float from interval [-1;1]
    """
    def __init__(self,value=1):
        self.value=value
        self.bias=np.random.uniform(-1,1)

        
class Synapse(object):
    """
    startNeuron, endNeuron: Neuron
    weight: float from interval [-1,1]
    """
    def __init__(self,startNeuron,endNeuron):
        self.startNeuron=startNeuron
        self.endNeuron=endNeuron
        self.weight=np.random.uniform(-1,1)


class NeuralNetwork(object):
    """
    layers: list of lists of Neuron
    synapses: list of lists of lists of Synapse
    weights: list of lists of lists of float
    learningrate: float
    dataset: Dataset
    """
    def __init__(self,dataset):
        self.layers=[]
        self.synapses=[]
        self.weights=[]
        self.learningrate=0 
        self.dataset=dataset
        self.predictions=[]
        self.accuracy=0
        #used for updateSynapseWeightMatrix
        self.matrixFrame=[]


    def createNetwork(self,length,layerSizes,learningrate):
        """
        creates a new network
        
        inputs:
        length: int
        layerSizes: list of int; len(layerSizes)==length
        learningrate: float
        """
        layers=[]
        for i in range(0,length):
            layers.append([])

        layerIndex=-1
        for layer in layers:
            layerIndex+=1
            for i in range(0,layerSizes[layerIndex]):
                layers[layerIndex].append(Neuron())
                
        synapses=[]
        matrixFrame=[]
        for i in range(0,len(layers)-1):
            synapses.append([])
            matrixFrame.append([])
        
        layerIndex=-1
        for layer in layers[:-1]:
            layerIndex+=1
            for nextNeuron in layers[layerIndex+1]:
                neuronSynapses=[]
                weightFrame=[]
                for neuron in layers[layerIndex]:
                    synapse=Synapse(neuron,nextNeuron)
                    neuronSynapses.append(synapse)
                    weightFrame.append(0)

                neuronSynapses=np.array(neuronSynapses)
                synapses[layerIndex].append(neuronSynapses)
                matrixFrame[layerIndex].append(weightFrame)

        self.layers=layers
        self.synapses=synapses
        self.matrixFrame=matrixFrame
        self.learningrate=learningrate
        self.updateSynapseWeightMatrix()

  
    def feedforward(self,data):
        """
        feeds a dataset through the network, returns activation values of last layer
        
        input: data: array of int (0<x<1)
        output: outputValues: array of int(0<x<1)
        """
        #initialise Input-Layer
        for neuron in self.layers[0]:
            index = self.layers[0].index(neuron)
            neuron.value=data[index]

        #feedforward
        layerIndex = 1
        
        while layerIndex < len(self.layers):
            inputMatrix = []
            
            for neuron in self.layers[layerIndex-1]:
                inputMatrix.append(neuron.value)
            neuronIndex = 0
            
            for neuron in self.layers[layerIndex]:
                endNeuronBias = neuron.bias
                
                weightMatrixRow = self.weights[layerIndex-1][neuronIndex]
                newValue = np.matmul(weightMatrixRow,inputMatrix)
                
                neuron.value = sigmoid(newValue + endNeuronBias)
                neuronIndex += 1
                    
            layerIndex += 1
        
        outputLayer = self.layers[-1]
        
        outputValues = []
        for outputNeuron in outputLayer:
            outputValues.append(outputNeuron.value)
        outputValues = np.array(outputValues)
        
        return outputValues

            
    def backpropagate(self,output,answer):
        """"
        corrects weights based on output of the network and desired output (answer)

        output, answer: array of float
        len(output)==len(answer)
        """
        #begins at output
        #calculate output error
        outputMatrix = output
        answerMatrix = answer
        errorMatrix = outputMatrix - answerMatrix

        weightMatrix = self.weights
        layerIndex = 0
        layerCounter = -1
        currentSynapseMatrix = np.array([])
        currentWeightMatrix = np.array([])
        transposedWeightMatrix = np.array([])
        synapses = self.synapses
        synapseWeights = self.weights

        #then goes from second last layer to first layer
        for layer in self.layers[:-1]:
            layerCounter += 1
            layerIndex = len(self.layers) -2 - layerCounter
            
            currentSynapseMatrix = synapses[layerIndex]
            currentWeightMatrix = synapseWeights[layerIndex]
            
            previousErrorMatrix = np.copy(errorMatrix)
            previousErrorMatrix = np.reshape(previousErrorMatrix,(len(previousErrorMatrix),1))
            transposedWeightMatrix = np.transpose(currentWeightMatrix)

            outputMatrix = []
            for neuron in self.layers[layerIndex+1]:
                outputMatrix.append(neuron.value)
            outputMatrix=np.array(outputMatrix)

            inputMatrix = []
            for neuron in self.layers[layerIndex]:
                inputMatrix.append(neuron.value)
            inputMatrix = np.array(inputMatrix)
            
            transposedInputMatrix = np.reshape(inputMatrix,(len(inputMatrix),1))

            errorMatrix = errorMatrix.flatten()

            deltaWeights = self.learningrate * errorMatrix * (outputMatrix*(1-outputMatrix)) * transposedInputMatrix

            #apply changes
            transposedCurrentSynapseMatrix = np.transpose(currentSynapseMatrix)
            arrayIndex = -1
            
            for changeValueArray in deltaWeights:
                arrayIndex += 1
                valueIndex =- 1
                
                for changeValue in changeValueArray:
                    valueIndex+=1
                    currentSynapse = transposedCurrentSynapseMatrix[arrayIndex][valueIndex]
                    currentChangeValue = deltaWeights[arrayIndex][valueIndex]

                    currentSynapse.weight -= currentChangeValue

            errorMatrix = np.matmul(transposedWeightMatrix,previousErrorMatrix)

        self.updateSynapseWeightMatrix()


    def getBiasMatrix(self):
        biasMatrix=[]
        for layer in self.layers:
            biasMatrix.append([])
            
        layerIndex=-1
        for layer in self.layers:
            layerIndex+=1
            for neuron in layer:
                biasMatrix[layerIndex].append(neuron.bias)

        return biasMatrix

                
    def updateSynapseWeightMatrix(self):
        """
        changes the weights in self.weights to the current values
        """
        weightMatrix=self.matrixFrame[:]
        layerIndex=-1
        for layerList in weightMatrix:
            layerIndex+=1
            rowIndex=-1
            for row in layerList:
                rowIndex+=1
                valueIndex=-1
                for value in row:
                    valueIndex+=1
                    synapse=self.synapses[layerIndex][rowIndex][valueIndex]
                    weightMatrix[layerIndex][rowIndex][valueIndex]=synapse.weight
                    
        self.weights=weightMatrix

        
    def updateSynapses(self):
        layerIndex=-1
        for layer in self.synapses:
            layerIndex+=1
            rowIndex=-1
            for row in layer:
                rowIndex+=1
                synapseIndex=-1
                for synapse in row:
                    synapseIndex+=1
                    synapse.weight=self.weights[layerIndex][rowIndex][synapseIndex]


    def updateBiases(self,biases):
        layerIndex=-1
        for layer in self.layers:
            layerIndex+=1
            neuronIndex=-1
            for neuron in layer:
                neuronIndex+=1
                neuron.bias=biases[layerIndex][neuronIndex]


    def predict(self,data):
        """
        receives data, returns prediction
    
        input: data: array of int (0<x<1)
        output: guess: int 
        """
        output=self.feedforward(data)
        guess=output.argmax(axis=0)
        
        return guess

    
    def train(self,setName,startIndex,endIndex): 
        """
        runs backpropagate for an interval of choice, saves predictions for later use

        setName: str ("train" or "test")
        startIndex, endIndex: int
        """
        if setName=="train":
            x_data= self.dataset.x_train
            y_data=self.dataset.y_train
        if setName=="test":
            x_data= self.dataset.x_test
            y_Data=self.dataset.y_test

        predictions=[]

        currentIndex=startIndex-1
        for data in x_data[startIndex:endIndex]:
            currentIndex+=1
            print("current Index: ",currentIndex)
            
            currentData=x_data[currentIndex]
            output=self.feedforward(data)
            answer=y_data[currentIndex]
            prediction=np.array(output)

            #normalise output for comparison with labels
            normalisedprediction=np.array([0.,0,0,0,0,0,0,0,0,0])
            argmaxprediction=np.argmax(prediction,0)
            normalisedprediction[argmaxprediction]=1
            predictions.append(normalisedprediction)

            self.backpropagate(output,answer)

        #if network is trained the first time
        if len(self.predictions) == 0:
            self.predictions.append([])
            
        for prediction in predictions:
            self.predictions[0].append(prediction)


    def updateAccuracy(self,setName):
        """
        calculates (self.)accuracy based on saved predictions

        setName: str ("train" or "test")
        """
        predictions=self.predictions[0]
        
        if setName=="train":
            answers=self.dataset.y_train[:len(predictions)]
        else:
            answers=self.dataset.y_test[:len(predictions)]

        predictionsArgmax = np.argmax(predictions,1)
        answersArgmax = np.argmax(answers,1)

        accuracy=(100.0 * np.sum(predictionsArgmax == answersArgmax) / len(predictions))

        self.accuracy=accuracy


    def calculateAccuracy(self,setName,startIndex,endIndex):
        """
        calculates and returns accuracy for a certain amount of data by running feedforward on every entry

        setName: str
        startIndex, endIndex: int
        """
        if setName=="train":
            x_data= self.dataset.x_train
            y_data=self.dataset.y_train
        if setName=="test":
            x_data= self.dataset.x_test
            y_data=self.dataset.y_test

        currentIndex=-1
        predictions=[]
        for data in x_data[startIndex:endIndex]:
            currentIndex+=1
            print("current Index: ",currentIndex)
            output=self.feedforward(data)
            predictions.append(output)

        answers=y_data[:len(predictions)]

        predictionsArgmax = np.argmax(predictions,1)
        answersArgmax = np.argmax(answers,1)
        accuracy=(100.0 * np.sum(predictionsArgmax == answersArgmax) / len(predictions))

        return accuracy

        
    def saveNetwork(self,name,directoryNames=["gespeicherte Netze"]):
        """
        saves network to file

        name: str
        directoryNames: list of str
        """        
        biasMatrix=self.getBiasMatrix()
        weightMatrix=self.weights
        config=[ [], [], [] , [] ]
        
        config[0].append(len(self.layers))
        for layer in self.layers:
            config[1].append(len(layer))
        config[2].append(self.learningrate)
        config[3].append(self.predictions)

        biasMatrix=np.array(biasMatrix)
        weightMatrix=np.array(weightMatrix)
        config=np.array(config)

        biasPath=getFilePath(directoryNames, name+"_BIASES")
        weightPath=getFilePath(directoryNames, name+"_WEIGHTS")
        configPath=getFilePath(directoryNames, name+"_CONFIG")
        
        np.save(biasPath, biasMatrix)
        np.save(weightPath , weightMatrix)
        np.save(configPath , config)


    def loadNetwork(self,name,directoryNames=["gespeicherte Netze"]):
        """
        loads network from file

        name: str
        directoryNames: list of str
        """
        biasPath=getFilePath(directoryNames,name+"_BIASES.npy")
        weightPath=getFilePath(directoryNames,name+"_WEIGHTS.npy")
        configPath=getFilePath(directoryNames,name+"_CONFIG.npy")
        
        biasMatrix= np.load(biasPath)
        weightMatrix= np.load(weightPath)
        config=np.load(configPath)

        biasMatrix=list(biasMatrix)
        weightMatrix=list(weightMatrix)
        config=list(config)

        self.createNetwork(config[0][0],config[1],config[2][0])

        self.weights=weightMatrix
        self.predictions=config[3][0]
        
        self.updateSynapses()
        self.updateBiases(biasMatrix)
        

    def plotAccuracyGraph(self,predictions=None,answers=None,points=100):
        """
        plots a graph based on predictions and answers (accuracy over time)

        predictions: list of array of int(0<x<1)
        answers: array of arrays of int (0<x<1)
        points: int; sets the amount of points in the graph
        """
        #set default values if needed
        if predictions==None:
            predictions=self.predictions[0]
        if answers==None:
            answers=self.dataset.y_train
            
        if points>len(predictions):
            points=len(predictions)
            
        accuracyOverTime=[0]

        indexValues=np.linspace( 0 , len(predictions) ,points)
        indexValues=indexValues.astype("int")

        for value in indexValues[1:]:
            currentPredictionsArgmax = np.argmax(predictions[:value],1)
            currentAnswersArgmax = np.argmax(answers[:value],1)
            
            currentAccuracy=(100.0 * np.sum(currentPredictionsArgmax == currentAnswersArgmax) / len(predictions[:value]))
            accuracyOverTime.append(currentAccuracy)
        
        plt.plot(indexValues,accuracyOverTime)
        plt.show()

    def ownImage(self):
        """
        opens an interface that allows the user to draw (a digit),
        which is then saved as a .png file,
        is then fed through the network
        and the prediction of the network is printed
        """
        drawImage()
        imageData=np.load("OwnImageData.npy")
        showImage()
        print(nn.predict(imageData))
        

if __name__=="__main__":
    
    #example: loading the MNIST database
    mnist=loadMNIST()
    mnistSet=Dataset(mnist, x_train=mnist[0],
                        x_test=mnist[1],
                        y_train=mnist[4],
                        y_test=mnist[5])


    #example: creating a network
    layerSizes=np.linspace(784,10,num=4) #28*28=784
    layerSizes=layerSizes.astype("int")
    nn=NeuralNetwork(mnistSet)
    nn.createNetwork(4,layerSizes,0.1)


    #example: training the network
    nn.train("train",0,10)
    nn.updateAccuracy("train")


    #example: loading a network
    nn.loadNetwork("784-526-268-10--01--60000")


    #example: graphical visualisation: accuracy over time
    nn.plotAccuracyGraph()

    #example: drawing an own image and feeding it through the network
    nn.ownImage()


import numpy as np

#nitialize a random seed, so that you can reproduce 
#the same results when running the program again
np.random.seed(444)


#import the keras objects that will be  used
# to build the neural network.
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD


#x array containing 4 possible A-B sets of inputs 
# for the XOR operation
X=np.array([[0,0],[0,1],[1,0],[1,1]])


#y array containing outputs for each possible
# set of inputs defined in x
y=np.array([[0],[1],[1],[0]])


#---------------------------------------------------------------------
#now we define the neural network

#The Sequential() model is one of the models provided
#by Keras to define a neural network.

#In Seq(),Layers of the network are defined
#in a sequential way
model=Sequential()



#First later of neurons, composed of 2 neaurons
# and fed by 2 inputs,
#defining their activation function as sigmoid function
model.add(Dense(2,input_dim=2))
model.add(Activation('sigmoid'))



#2nd(output layer of neurons),composed of one neuron
#and activation function-sigmoid
model.add(Dense(1))
model.add(Activation('sigmoid'))


#-------------------------------------------------------------
#Details about training the network-


#to adjust the weights of the network, we use the Scholastic Gradient Descent (SGD)
#with learning rate,lr=0.1
sgd=SGD(lr=0.1)


#we use the mean squared error as a loss function to be minimized.
model.compile(loss='mean_squared_error', optimizer=sgd)


#------------------------------------------------------------
#Training the network-

#perform the training by running the fit() method,
# using X and y as training examples 
#and updating the weights after every training example is fed into the network (batch_size=1). 
#The number of epochs represents the number of times the whole training set will be used to train the neural network.
model.fit(X, y, batch_size=1, epochs=5000)


#------------------------------------------------
if __name__ == '__main__':
    print(model.predict(X))


 #loss in reduced each time the model is fed with new training examples (and this is done 5000 times)

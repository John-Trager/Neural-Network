"""
nn demo
"""
import numpy as np

#debug when True will print to console extra info else will only print result 
debug = True

def sigmoid(x):
    """
    Method takes in integer or float and returns a value
    between 0 and 1
    """
    return 1 / (1 + np.exp(-x))

def sigmoid_derivitive(x):
    """
    Method takes in integer or float and returns the derivitive of the sigmoid of the number
    """
    return x * (1 - x)

training_inputs = np.array([[0,0],
                           [1,0],
                           [1,1],
                           [0,1]]) 

training_outputs = [[0],
                    [1],
                    [1],
                    [0]] 

#generates random numbers from a seed so same numbers evrytime we run the program
np.random.seed(1)

"""
np.random.random((x,y)) --> creates random array of x rows by y columns 

5 * np.random.random_sample((3, 2)) - 5
array([[-3.99149989, -0.52338984],
       [-2.99091858, -0.79479508],
       [-1.23204345, -1.75224494]])
"""
synp_weights = 2 * np.random.random((2,1)) - 1 #creating 2 by 1 array for weights [[w1],[w2]]

print('Random starting synaptic weights: ') if debug == True else None
print(synp_weights) if debug == True else None

"""
for loop runs the nural net and back-propogates in the process 
"""

for i in range(100000):

    input_layer = training_inputs

    """
    For 2-D arrays it is the matrix product:
        a = [[1, 0], [0, 1]]
        b = [[4, 1], [2, 2]]
        np.dot(a, b)
        array([[4, 1],
               [2, 2]])
    """
    #mutiplying input layer values ([a,b]) by the synaptic weights respectively ([w1,w2])
    #for all sets of inputs to give all the corosponding outputs
    outputs = sigmoid(np.dot(input_layer, synp_weights))

    #error -> the difference in the calculated output to the actual output
    #done by subtracting the list of training outputs by the list of generated outputs 
    #([a1,b1]) - ([ga1,gb1]) for all values
    error = training_outputs - outputs

    #the adjustment is a scalar value that is porpotional to the error * the sigmoid derivitive of the list of generated outputs
    #adjustment is used to calculate the amount of change of the weights during back propogartion
    adjustment = error * sigmoid_derivitive(outputs)

    #updating the synaptic weights via backpropogation
    #The transpose of a matrix is a new matrix whose rows are the columns of the original
    #transposing matrix in this case is used to transform the matrixs columns ([],[],[])
    # into rows ([],
    #            [],     *   scalar --> new synaptic weight values
    #            [])
    #so that you can multiply all the values in the matrix by a scalar in this case the adjustment
    synp_weights += np.dot(input_layer.T, adjustment)

#printing results
print('Weights after training: ')
print(synp_weights)

print('Outputs after training: ')
print(outputs)
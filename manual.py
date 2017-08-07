import numpy as np
import sklearn.datasets
import pandas as pd
import matplotlib.pyplot as plt

def get_moon():
    np.random.seed(0)
    data = sklearn.datasets.make_moons(300, noise= 0.3)
    
    result = pd.DataFrame(data = data[0], columns = ['x', 'y'])
    result['color'] = data[1]
    
    return result 

def plot_decision_boundary(pred_func, data ):
    
    # Set min and max values and give it some padding
    x_min, x_max = data['x'].min() - .5, data['x'].max() + .5
    y_min, y_max = data['y'].min() - .5, data['y'].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(data['x'], data['y'], c= data['color'], cmap=plt.cm.Spectral)



    
def tanh(data):
    return np.tanh(data)

def sigmoid(data):
    return 1/(1 + pow(np.e, -data))

def get_error(prediction, expected):
    result = expected - prediction
    result = (result * result)/2.0
    return np.sum(result)

def derivative(result):
    return  result

def predict( input_data ):
    return tanh(np.dot(input_data, weights )).sum(1)



data = get_moon()

size = 3
chunk_size = 5
lr = 0.001
iterations = 1

x = data[['x', 'y']].values
y = data['color'].values


weights = pd.DataFrame(np.random.random([x.shape[1], size]).T, columns = ['x', 'y'])


for num in range(0, iterations):
    observation = data.sample(1)
    x = observation[['x', 'y']].values
    y = observation['color'].values
    
        

        
    
    
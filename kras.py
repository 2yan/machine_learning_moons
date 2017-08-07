import numpy as np
import pandas as pd
import sklearn.datasets
import matplotlib.pyplot as plt
import keras 
from keras.layers import Dense
from keras.layers import Activation
from keras.optimizers import Adam, SGD

        
        

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

def get_predictor(data):
    size = 50
    x = data[['x', 'y']].values
    y = data['color'].values
    
    model = keras.models.Sequential()
    model.add(Dense(size, input_shape=(2,), name = 'input'))
    model.add(Dense(size, activation = keras.activations.tanh, name = 'hidden' ) )
    
    model.add(Dense(1, name ='output'))
    model.add(Activation("sigmoid", name = 'final'))    
    
    opti = SGD(lr = 0.1, decay = 0.001)
    model.compile(loss="binary_crossentropy", optimizer= opti,
    	metrics=["accuracy"])
    model.fit(x, y, batch_size = 1, epochs = 200)
    
    return model




data = get_moon()


model = get_predictor(data)
func = model.predict
plot_decision_boundary(func, data)
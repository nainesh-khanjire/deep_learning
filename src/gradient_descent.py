import numpy as np
import matplotlib.pyplot as plt

# gradient descent is a optimization algorithm to find 
# min or atleast local minimum of a function and uses machine learning to
# to optimize weights and biases

def y_funtion(x):
    return x**2 

def y_derivative(x):
    return 2*x


x = np.arange(-100,100,0.1)
y = y_funtion(x)

current_posn = (80,y_funtion(80))
learning_rate = 0.01



for _ in range(1000):
    new_x = current_posn[0] - learning_rate * y_derivative(current_posn[0])
    new_y = y_funtion(new_x)
    current_posn = (new_x,new_y)
    plt.plot(x,y)
    plt.scatter(current_posn[0],current_posn[1],color = 'red')
    plt.pause(0.001)
    plt.clf()


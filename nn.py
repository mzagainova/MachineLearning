import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib

np.random.seed(0)
X, y = sklearn.datasets.make_moons(200, noise=0.20)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
num_examples = len(X)

def derLdy(y, y_hat):
    global num_examples
    dy = y_hat
    dy[range(num_examples),y] -= 1
    dy /= num_examples
    return dy

def derLda(a, dLdy, W2):
    return ((1 - (np.tanh(a)**2)) * (dLdy.dot(np.ndarray.transpose(W2))))

def derLdW2(h, dLdy):
    return np.ndarray.transpose(h).dot(dLdy)

def derLdW1(x, dLda):
    return np.ndarray.transpose(x).dot(dLda)

def softmax(z):
    exps = np.exp(z)
    softmax = exps / np.sum(exps, axis=1, keepdims=True)
    return softmax

def calc_model(model, X):
    # calculate prediction
    a = X.dot(model['W1']) + model['b1']
    h = np.tanh(a)
    z = h.dot(model['W2']) + model['b2']
    y_hat = softmax(z)
    model_params = {'a' : a, 'h' : h, 'z' : z, 'y_hat': y_hat}

    return model_params

# model is current version of the model, dictionary ('W1'=W1,'b1'=b1,etc.)
# X is all training data
# y is training labels
def calculate_loss(model, X, y):
    global num_examples
    params = calc_model(model, X)
    corect_logprobs = -np.log(params['y_hat'][range(num_examples), y])
    data_loss = np.sum(corect_logprobs)
    loss = (1 / float(num_examples)) * data_loss
    return loss

# helper function to predict an output (0 or 1)
# model is dictionary
# x is one sample, without label
def predict(model, x):
    params = calc_model(model, x)
    y_hat = params['y_hat']
    #print y_hat
    #print "Y HAT: ", np.argmax(y_hat, axis=1)
    return np.argmax(y_hat, axis=1)

def calc_grads(model, model_params, y, x):
    dLdy = derLdy(y, model_params['y_hat'])
    dLdW2 = derLdW2(model_params['h'], dLdy)
    dLdb2 = np.sum(dLdy, axis=0, keepdims=True)
    dLda = derLda(model_params['a'], dLdy, model['W2'])
    dLdW1 = derLdW1(x, dLda)
    dLdb1 = np.sum(dLda, axis=0)

    gradients = {'dLdy':dLdy, 'dLdW2':dLdW2, 'dLdb2':dLdb2, 'dLda':dLda,
    'dLdW1':dLdW1, 'dLdb1':dLdb1}
    return gradients

def update_model(model, gradients):
    learning_rate = 0.1
    model['W1'] -= learning_rate * gradients['dLdW1']
    model['b1'] -= learning_rate * gradients['dLdb1']
    model['W2'] -= learning_rate * gradients['dLdW2']
    model['b2'] -= learning_rate * gradients['dLdb2']
    return model

# this learn parameters for the neural network and returns the model
# X is the training data
# y is the labels
# nn_hdim: number of nodes in the hidden layer
# num_passes: number of passes through the training data for gradient descent
# print_loss: if true, pring the loss every 1000 iterations
def build_model(X, y, nn_hdim, num_passes=20000, print_loss=False):
    # initialize W and b
    W1 = np.random.randn(2, nn_hdim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, 2)
    b2 = np.zeros((1, 2))
    model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    # run for number of epochs
    for i in range(0, num_passes):
        model_params = calc_model(model, X)
        gradients = calc_grads(model, model_params, y, X)
        model = update_model(model, gradients)
        if print_loss and i % 1000 == 0:
            print("Loss at iteration ", i, ": ", calculate_loss(model, X, y))
    return model

def plot_decision_boundary (pred_func):
    # Set min and max values and give it some padding 2
    x_min , x_max = X[:, 0] .min() - .5, X[:,0].max() + .5
    y_min , y_max = X[:, 1] .min() - .5, X[:,1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min,x_max,h) , np.arange(y_min,y_max,h))

    # Predict function value for whole grid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # plot countour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

plt.figure(figsize=(6, 14))
hidden_layer_dimensions = [1,2,3,4,5]
for i , nn_hdim in enumerate(hidden_layer_dimensions):
    plt.subplot(5 ,2 ,i +1)
    plt.title('Hidden Layer Size %d' % nn_hdim)
    model = build_model(X, y, nn_hdim, print_loss=True)
    plot_decision_boundary(lambda x : predict(model, x))
plt.show()

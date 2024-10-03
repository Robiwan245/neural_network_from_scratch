import numpy as np
from data import load_data
from tqdm import tqdm

class NeuralNetwork:
    def __init__(
            self,
            train, 
            labels,
            n_layers = 2,
            layers_size = [784, 10, 10],
            batch_size = 1024,
            lr = 0.1
            ):
        
        self.train = train
        self.labels = labels

        self.N = len(self.labels)

        self.batch_size = batch_size
        self.lr = lr

        self.n_layers = n_layers
        self.layers_size = layers_size

        self.parameters = {}
        self.initialize_parameters()

    def initialize_parameters(self):
        for layer in range(1, self.n_layers + 1):
            self.parameters[f'W{layer}'] = np.random.rand(
                self.layers_size[layer], 
                self.layers_size[layer - 1]
            ) / np.sqrt(self.layers_size[layer - 1])
            self.parameters[f'b{layer}'] = np.zeros((self.layers_size[layer], 1))

    def cross_entropy(self, labels, predictions):
        N = len(labels)
        return -(1 / N) * np.sum(labels * np.log(predictions))
        
    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=0))
        return e_x / np.sum(e_x, axis=0)
    
    def relu(self, x):
        x[x < 0] = 0
        return x
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def forward(self, x):
        self.states = {}

        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])

        a = x.T
        self.states[f'A0'] = a

        for layer in tqdm(range(self.n_layers - 1)):
            z = np.dot(self.parameters[f'W{layer + 1}'], a) + self.parameters[f'b{layer + 1}']
            a = self.sigmoid(z)
            
            self.states[f'A{layer + 1}'] = a
            self.states[f'Z{layer + 1}'] = z
            self.states[f'W{layer + 1}'] = self.parameters[f'W{layer + 1}']

        z = np.dot(self.parameters[f'W{self.n_layers}'], a) + self.parameters[f'b{self.n_layers}']
        out = self.softmax(z)

        self.states[f'A{self.n_layers}'] = out
        self.states[f'Z{self.n_layers}'] = z
        self.states[f'W{self.n_layers}'] = self.parameters[f'W{self.n_layers}']

        return out
    
    def backprop(self, out, labels):

        self.derivatives = {}

        dZ = out - labels
        
        dW = np.dot(dZ, self.states[f'A{self.n_layers}'].T) / self.batch_size # dJ(W) / dw2 = dJ(W) / dy_hat * dy_hat / dw2
        db = np.sum(dZ, axis=1, keepdims=True) / self.batch_size
        dA = np.dot(self.states[f'W{self.n_layers}'].T, dZ)

        self.derivatives[f'dW{self.n_layers}'] = dW
        self.derivatives[f'db{self.n_layers}'] = db

        for layer in range(self.n_layers - 1, 0, -1):
            dZ = dA * self.sigmoid_derivative(self.states[f'Z{layer}'])
            dW = 1 / self.batch_size * np.dot(dZ, self.states[f'A{layer - 1}'].T)
            db = 1 / self.batch_size * np.sum(dZ, axis=1, keepdims=True)

            self.derivatives[f'dW{layer}'] = dW
            self.derivatives[f'db{layer}'] = db

            if layer > 1:
                dA = np.dot(self.states[f'W{layer}'].T, dZ)

    def update(self):
        for layer in range(1, self.n_layers + 1):
            self.parameters[f'W{layer}'] = self.parameters[f'W{layer}'] - self.lr * self.derivatives[f'dW{layer}']
            self.parameters[f'b{layer}'] = self.parameters[f'b{layer}'] - self.lr * self.derivatives[f'db{layer}']


def one_hot(y):
    one_hot_y = np.zeros((y.size, y.max() + 1))
    one_hot_y[np.arange(y.size), y] = 1
    return one_hot_y.T

(x_train, y_train), (x_test, y_test) = load_data()

x_train = x_train / 255
y_train = one_hot(y_train[:1028])

model = NeuralNetwork(x_train, y_train)

for i in range(100):
    out = model.forward(np.array(x_train[:1028]))

    acc = model.cross_entropy(y_train[:1028], out)
    print(acc)

    model.backprop(out, y_train[:1028])
    model.update()
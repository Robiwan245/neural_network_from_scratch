import numpy as np
from data import load_data
from tqdm import tqdm

class NeuralNetwork:
    def __init__(
            self,
            train, 
            labels,
            n_layers=2,
            layers_size = [784, 64, 10]
            ):
        
        self.train = train
        self.labels = labels

        self.N = len(self.labels)

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

    def effective_softmax_derivative(self, out):
        s = out.reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)
    
    def softmax_derivative(self, out):
        d_out = np.zeros((out.shape[0], out.shape[0]))
        
        for i in range(out.shape[0]):
            for j in range(out.shape[0]):
                if i == j:
                    d_out[i, j] = out[i,j] * (1 - out[i,j])
                else:
                    d_out[i, j] = -out[i,j] * out[i,j]

        return d_out
    
    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    
    def loss_derivative(self, labels, out):
        return out - labels

    def forward(self, x):
        self.states = {}

        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])

        a = x.T

        for layer in tqdm(range(self.n_layers - 1)):
            z = np.dot(self.parameters[f'W{layer + 1}'], a) + self.parameters[f'b{layer + 1}']
            a = self.sigmoid(z)
            
            self.states[f'A{layer + 1}'] = a
            self.states[f'Z{layer + 1}'] = z
            self.states[f'W{layer + 1}'] = self.parameters[f'W{layer + 1}']

        z = np.dot(self.parameters[f'W{self.n_layers}'], a) + self.parameters[f'b{self.n_layers}']
        out = self.softmax(z)

        self.states[f'A{self.n_layers}'] = a
        self.states[f'Z{self.n_layers}'] = out
        self.states[f'W{self.n_layers}'] = self.parameters[f'W{self.n_layers}']

        return out
    
    def backprop(self, out, labels):

        d_loss = self.loss_derivative(labels, out)
        d_out = self.effective_softmax_derivative(out)






(x_train, y_train), (x_test, y_test) = load_data()

model = NeuralNetwork(x_train, y_train)
out = model.forward(np.array(x_train[:1028]))

acc = model.cross_entropy(y_train[:1028], out)

model.backprop(y_train[:1028], out)
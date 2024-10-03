import numpy as np
from data import load_data
from tqdm import tqdm

class NeuralNetwork:
    def __init__(
            self,
            train, 
            labels,
            n_layers=5,
            layers_size = [28, 56, 112, 56, 28]
            ):
        
        self.train = train
        self.labels = labels

        self.N = len(self.labels)

        self.n_layers = n_layers
        self.layers_size = layers_size

        self.parameters = {}
        self.initilize_parameters()

    def initilize_parameters(self):
        for layer in range(1, self.n_layers):
            self.parameters[f'W{layer}'] = np.random.rand(
                self.layers_size[layer], 
                self.layers_size[layer - 1]
            ) / np.sqrt(self.layers_size[layer - 1])
            self.parameters[f'b{layer}'] = np.zeros((self.layers_size[layer], 1))

    def cross_entropy(self, predictions):
        return (1 / self.N) * np.sum(self.labels * np.log(predictions))
        
    def forward(self, x):
        states = {}

        for layer in tqdm(range(1, self.n_layers)):
            x = self.parameters[f'W{layer}'] @ x.T + self.parameters[f'b{layer}']

(x_train, y_train), (x_test, y_test) = load_data()

model = NeuralNetwork(x_train, y_train)
model.forward(x_train)
import numpy as np
from functions import softmax, tanh, tanh_derivative

class Layer:
    def __init__(self, nin, nout, is_output=False):
        self.W = np.random.randn(nin, nout) * np.sqrt(2. / nin)
        self.b = np.zeros((1, nout))
        self.is_output = is_output
        self.Inputs, self.Z, self.A = None, None, None
        
        # Adam Parameters
        self.mW, self.vW = np.zeros_like(self.W), np.zeros_like(self.W)
        self.mb, self.vb = np.zeros_like(self.b), np.zeros_like(self.b)

    def forward(self, inputs):
        self.Inputs = inputs
        self.Z = np.dot(inputs, self.W) + self.b
        self.A = softmax(self.Z) if self.is_output else tanh(self.Z)
        return self.A

class MLP:
    def __init__(self, layer_sizes):
        self.layers = []
        self.t = 0 # Global step for Adam
        for i in range(len(layer_sizes) - 1):
            is_out = (i == len(layer_sizes) - 2)
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1], is_output=is_out))

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, y_true, y_pred, learning_rate, opt):
        m = y_true.shape[0]
        dZ = y_pred - y_true
        self.t += 1
        
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            
            dW = np.dot(layer.Inputs.T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
            
            if i > 0:
                prev_layer = self.layers[i - 1]
                dA = np.dot(dZ, layer.W.T)
                dZ = dA * tanh_derivative(prev_layer.A)
            
            if opt == 'Adam':
                layer.mW = beta1 * layer.mW + (1 - beta1) * dW
                layer.vW = beta2 * layer.vW + (1 - beta2) * (dW ** 2)
                mWh = layer.mW / (1 - beta1 ** self.t)
                vWh = layer.vW / (1 - beta2 ** self.t)
                layer.W -= learning_rate * mWh / (np.sqrt(vWh) + epsilon)
                
                layer.mb = beta1 * layer.mb + (1 - beta1) * db
                layer.vb = beta2 * layer.vb + (1 - beta2) * (db ** 2)
                mbh = layer.mb / (1 - beta1 ** self.t)
                vbh = layer.vb / (1 - beta2 ** self.t)
                layer.b -= learning_rate * mbh / (np.sqrt(vbh) + epsilon)
            else: # Fallback to GD
                layer.W -= learning_rate * dW
                layer.b -= learning_rate * db
import torch


class Neuron():
    def __init__(self, nin) -> None:
        super().__init__()
        self.w = torch.randn(nin, requires_grad=True)
        self.b = torch.randn(1, requires_grad=True)

    def __call__(self, x):
        return torch.add(torch.matmul(x,self.w.unsqueeze(1)), self.b).relu()

    def parameters(self):
        return  torch.concat((self.b, self.w), 0).unsqueeze(0)

class Layer():
    def __init__(self, nin, nout) -> None:
        super().__init__()
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        return torch.Tensor([neuron(x) for neuron in self.neurons])
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP():
    def __init__(self, nin, nout):
        size = [nin] + nout
        self.layers = [Layer(size[i], size[i+1]) for i in range(len(nout))]
        self.parameters = self.parameters()

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x / x.sum()

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def train(self, X, Y, alpha=0.1, iter=100):
        for i in range(iter):
            probs = self.__call__(X)
            loss = -probs[Y.type(torch.int64)].log().mean()
            loss_alpha = alpha*loss
            self.parameters = torch.subtract(self.parameters, loss_alpha)
        return self.parameters


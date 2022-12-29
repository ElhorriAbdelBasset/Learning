import torch


class Neuron():
    def __init__(self, nin) -> None:
        super().__init__()
        self.w = torch.randn((nin,1), requires_grad=True)
        self.b = torch.randn(1, requires_grad=True)

    def __call__(self, x):
        return sum((wi*xi for wi, xi in zip(self.w,x)), self.b).relu()

    def parameters(self):
        return  self.w + [self.b]

class Layer():
    def __init__(self, nin, nout) -> None:
        super().__init__()
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        return [neuron(x) for neuron in self.neurons]
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP():
    def __init__(self, nin, nout):
        size = [nin] + nout
        self.layers = [Layer(size[i], size[i+1]) for i in range(len(nout))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
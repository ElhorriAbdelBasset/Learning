import torch


class Neuron():
    def __init__(self, nin) -> None:
        super().__init__()
        self.w = torch.randn(nin, requires_grad=True)               # w = [[x1, x2, x3 .... xn]]    (n,1)
        self.b = torch.randn(1, requires_grad=True)

    def __call__(self, x):
        return torch.add(torch.dot(x,self.w.T), self.b).relu()      # x = [[x1, x2, x3 .... xn]]    (1,n)

    def parameters(self):
        return  torch.concat((self.b, self.w), 0).unsqueeze(0)

class Layer():
    def __init__(self, nin, nout):
        super().__init__()
        self.neurons = [Neuron(nin) for _ in range(nout)]
        self.parameters = self.parameters()

    def __call__(self, x):
        x = torch.concat((torch.ones(size=(1,x.shape[1])), x), 0)
        z = self.parameters @ x
        return z
    
    def parameters(self):
        return torch.concat([neuron.parameters() for neuron in self.neurons], dim=0)

class MLP():
    def __init__(self, nin, nout):
        size = [nin] + nout
        self.layers = [Layer(size[i], size[i+1]) for i in range(len(nout))]
        self.parameters = self.parameters()
        self.activations = {}

    def __call__(self, x):
        for layer in range(len(self.layers)-1):
            z = self.layers[layer](x)
            x = torch.relu(z)
            self.activations[layer] = {"z":z, "a":x}
        z = self.layers[len(self.layers)-1](x)
        x = ((torch.sigmoid(-z) - 0.5).ceil()).abs()
        self.activations[layer+1] = {"z":z, "a":torch.sigmoid(-z)}
        return x

    def parameters(self):
        return [layer.parameters for layer in self.layers]

    def train(self, X, Y, alpha=0.1, iter=100):
        for _ in range(iter):
            probs = self.__call__(X)
            loss = (probs.reshape((-1,1))[Y.type(torch.int64)].squeeze(1)).log().mean()
            loss_alpha = alpha*loss

        d_sigmoid = self.activations[2]["a"]*(1-self.activations[2]["a"])

        print(loss_alpha)

        return self.parameters

    def test(self,X, Y):
        probs = self.__call__(X)
        cats = probs >= 0.5
        acc = (cats == Y.reshape(-1)).sum()
        acc =  float(acc) / Y.shape[0]
        print(acc)


import random
from engine import Value

class Neuron:
    """
                     ········                       
      weights in     :NEURON:                       
                     ········                       
         W0                                         
         XXX            XXXXXXX                     
            XXXX      XX       XX             XX    
         Wk     XXXX X           X tanh(out)    XX  
          XXXXXXXXXXXX   bias    XXXXXXXXXXXXXXXXXX 
                XXX  X           X              XX  
        Wnin XXX      X        XX             XX    
           XX           XXXXXXX                     

    """
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        zip(self.w, x)
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return (self.w + [self.b])

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return (outs)

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return (x)

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def train(self, xs, ys, n_iter, learning_rate):
        for k in range(n_iter):
            ypred = [n(x) for x in xs]
            ypred = [x[0] for x in ypred]
            loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))
            for p in n.parameters():
                p.grad =- 0.0
            loss.backward()
            for p in n.parameters():
                p.data += -learning_rate * p.grad
            print (k, loss.data)

n = MLP(3 , [4, 4, 1])

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]

ys = [1.0, -1.0, -1.0, 1.0]

n.train(xs, ys, 40, 0.1)

print(str(ys))

for x in xs:
    print(n(x))

print (str(n.parameters()))

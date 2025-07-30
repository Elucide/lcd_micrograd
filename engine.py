import math
import matplotlib as plt


class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self._prev = set(_children)
        self.op = _op
        self.grad = 0
        self._backward = lambda: None

    def __repr__(self):
        return (f"value(data={self.data})")

    def __add__(self, right):
        out = Value(self.data + right.data, (self, right), '+')

        def _backward():
            self.grad = 1.0 * out.grad
            other.grad = 1.0 * out.grad
        
        out._backward = _backward
        return (out)

    def __mul__(self, right):
        out = Value (self.data * right.data, (self, right), '*')
        def _backward():
            self.grad = other.data * out.grad
            other.grad = self.data * out.grad

        out._backward = _backward
        return (out)

    def tanh(self):
        n = self.data
        t = (math.exp(2 * n) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self, ), "tanh")

        def _background():
            self.grad = out.grad = (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1
        for v in reverse(topo):
            v._backward()


a = Value(2.0)
b = Value(-3.0)
print(a)
print(b)
print(a + b)

import math
import matplotlib as plt

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self._prev = set(_children)
        self.op = _op
        self.grad = 0.
        self._backward = lambda: None

    def __repr__(self):
        return (f"value(data={self.data})")

    def __add__(self, right):
        right = right if isinstance(right, Value) else Value(right)
        out = Value(self.data + right.data, (self, right), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            right.grad += 1.0 * out.grad
        
        out._backward = _backward
        return (out)

    def __radd__(self, right):
        return (self + right)

    def __mul__(self, right):
        right = right if isinstance(right, Value) else Value(right)
        out = Value (self.data * right.data, (self, right), '*')
        def _backward():
            self.grad += right.data * out.grad
            right.grad += self.data * out.grad

        out._backward = _backward
        return (out)

    def __rmul__(self, right):
        return (self * right)

    def __pow__(self, right):
        assert isinstance(right, (int, float))
        out = Value(self.data**right, (self,), f'**{right}')

        def _backward():
            self.grad += right * (self.data ** (right -1)) * out.grad
        out._backward = _backward

        return out

    def __truediv__(this, right):
        return self * right ** -1

    def __neg__(self):
        return self * -1

    def __sub__(self, right):
        return self + (-right)

    def tanh(self):
        n = self.data
        t = (math.exp(2 * n) - 1) / (math.exp(2 * n) + 1)
        out = Value(t, (self, ), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        n = self.data
        out = Value(math.exp(this.data), (self, ), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1
        for v in reversed(topo):
            v._backward()


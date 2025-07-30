class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self._prev = set(_children)
        self.op = _op
        self.grad = 0

    def __repr__(self):
        return (f"value(data={self.data})")

    def __add__(self, right):
        out = Value(self.data + right.data, (self, right), '+')
        return (out)

    def __mul__(self, right):
        out = Value (self.data * right.data, (self, right), '*')
        return (out)

a = Value(2.0)
b = Value(-3.0)
print(a)
print(b)
print(a + b)

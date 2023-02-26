import operator


class OperatorTemplate:

    def _operator(self, operator, *other):
        raise NotImplementedError

    # --- Binary operators

    def __add__(self, other):
        return self._operator(operator.add, other)

    def __sub__(self, other):
        return self._operator(operator.sub, other)

    def __mul__(self, other):
        return self._operator(operator.mul, other)

    def __truediv__(self, other):
        return self._operator(operator.truediv, other)

    def __floordiv__(self, other):
        return self._operator(operator.floordiv, other)

    def __mod__(self, other):
        return self._operator(operator.mod, other)

    # Comparisons

    def __eq__(self, other):
        return self._operator(operator.eq, other)

    def __ne__(self, other):
        return self._operator(operator.ne, other)

    def __gt__(self, other):
        return self._operator(operator.gt, other)

    def __ge__(self, other):
        return self._operator(operator.ge, other)

    def __lt__(self, other):
        return self._operator(operator.lt, other)

    def __le__(self, other):
        return self._operator(operator.le, other)

    # --- Unary operators

    def __pos__(self):
        return self._operator(operator.pos)

    def __neg__(self):
        return self._operator(operator.neg)

    def __abs__(self):
        return self._operator(operator.abs)

from __future__ import annotations
import operator


class OperatorTemplate:

    def _operator(self, operator, *other, swap: bool = False) -> OperatorTemplate:
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

    def __pow__(self, other):
        return self._operator(operator.pow, other)

    # Right-sided

    def __radd__(self, other):
        return self._operator(operator.add, other, swap=True)

    def __rsub__(self, other):
        return self._operator(operator.sub, other, swap=True)

    def __rmul__(self, other):
        return self._operator(operator.mul, other, swap=True)

    def __rtruediv__(self, other):
        return self._operator(operator.truediv, other, swap=True)

    def __rfloordiv__(self, other):
        return self._operator(operator.floordiv, other, swap=True)

    def __rpow__(self, other):
        return self._operator(operator.pow, other, swap=True)

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

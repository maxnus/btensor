

class Space:

    __next_id = 0

    def __init__(self, size, parent=None):
        self._id = self.get_next_id()
        self._size = size
        self._parent = parent
        self._basis = tuple()
        self._root = parent.root if parent is not None else self

    def __repr__(self):
        return '%s(id= %d, size= %d)' % (type(self).__name__, self.id, self.size)

    @staticmethod
    def get_next_id():
        next_id = Space.__next_id
        Space.__next_id += 1
        return next_id

    @property
    def id(self):
        return self._id

    @property
    def size(self):
        return self._size

    @property
    def parent(self):
        return self._parent

    @property
    def basis(self):
        return self._basis

    @property
    def root(self):
        return self._root

    @property
    def nbasis(self):
        return len(self.basis)

    def add_basis(self, basis):
        self._basis = self._basis + (basis,)

    def is_root(self):
        return self.parent is None

    def has_same_root(self, other):
        return self.root == other.root

    def check_same_root(self, other):
        if self.has_same_root(other):
            return
        raise RuntimeError("Space %s and %s do not have the same root." % (self, other))

    def is_subspace_of(self, other, inclusive=True):
        """True if space is subspace of other."""
        return other in self.get_parents(include_self=inclusive)

    def is_superspace_of(self, other, inclusive=True):
        """True if space is superspace of other."""
        return other.is_superspace_of(self, inclusive=inclusive)

    def get_parents(self, include_root=True, include_self=False):
        """Get list of parent bases ordered from direct parent to root basis."""
        parents = []
        current = self
        if include_self:
            parents.append(current)
        while not current.is_root():
            parents.append(current.parent)
            current = current.parent
        if not include_root:
            parents = parents[:-1]
        return parents

    def find_common_parent(self, other):
        """Find lowest common ancestor between two bases."""
        if not self.has_same_root(other):
            return None
        parents1 = self.get_parents(include_self=True)[::-1]
        parents2 = other.get_parents(include_self=True)[::-1]
        assert (parents1[0] == parents2[0])
        parent = None
        for p1, p2 in zip(parents1, parents2):
            if p1 == p2:
                parent = p1
            else:
                break
        assert parent is not None
        return parent
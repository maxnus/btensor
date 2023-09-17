.. include:: links.rst

.. _quickstart:

==========
Quickstart
==========

As a first example, consider the 2D euclidian basis with basis vectors :math:`\mathbf{e}_x` and :math:`\mathbf{e}_y`,
and a second (non-orthogonal) basis, with basis vectors
:math:`\mathbf{e}_{x'}` =  :math:`\mathbf{e}_x` and
:math:`\mathbf{e}_{y'}` =  :math:`\frac{1}{\sqrt{2}} \left( \mathbf{e}_x + \mathbf{e}_y \right)`.
In other words, the first basis vector is identical, however the second basis vector is rotated 45° clockwise.

In BTensor, a :ref:`Basis <api/_autosummary/btensor.basis.Basis:Basis>` can be defined according to

.. code-block:: python

    import numpy as np
    from btensor import Basis

    basis1 = Basis(2)
    tm = np.asarray([[1, 1/np.sqrt(2)],
                     [0, 1/np.sqrt(2)]])
    basis2 = Basis(tm, parent=basis1)

where ``basis1`` represents the euclidian 2D basis, ``tm`` the transformation matrix, and ``basis2`` the second,
non-orthogonal basis.
Note that the definition of ``basis1`` is very simple: only an integer defining the dimensionality of the space is
required.
In contrast, ``basis2`` is defined in terms of a transformation matrix and a parent basis, namely ``basis1``.
Note, that the :ref:`make_subbasis <api/_autosummary/btensor.basis.Basis.make_subbasis:Basis.make\\_subbasis>` method of
``basis1`` could have been used instead.

In BTensor, bases are organized in a *tree structure*. We distinguish two types of bases:

- A **root-basis** does not have a parent and is constructed from an integer size argument.
- A **derived basis** has a parent basis and is defined in terms of a transformation wrt to its parent.

In this example, ``basis1`` is a root basis and ``basis2`` is a derived basis.

.. note::

    The root-basis is not required to be orthogonal.
    A non-orthogonal root basis can be constructed as ``Basis(2, metric=m)``, where ``m`` is the metric matrix of
    the root-basis.

All bases which belong to the same basis tree are considered **compatible**, i.e., BTensor can perform numerical
operations such as addition between tensors expressed in these bases.


For this we require the second fundamental type, the :ref:`Tensor <api/_autosummary/btensor.tensor.Tensor:Tensor>`,
which wraps NumPy's ndarray_.
Let us consider the points :math:`\mathbf{p}_1 = -1\mathbf{e}_{x} + 1\mathbf{e}_{y}` and
:math:`\mathbf{p}_2 = 1\mathbf{e}_{x'} + 1\mathbf{e}_{y'}`.
We can construct these as follows:

.. code-block:: python

    from btensor import Tensor
    point1 = Tensor([-1, 0], basis=basis1)
    point2 = Tensor([ 1, 1], basis=basis2)

The important thing to note is that the representations :math:`(-1, 0)` and :math:`(1, 1)` of these two points refer to
differents bases. In particular, it does not make sense to add these representations directly, but instead we have to consider the basis vectors they represent, such that
:math:`\mathbf{p}_3 = \mathbf{p}_1 + \mathbf{p}_2 = -1\mathbf{e}_{x} + 1\mathbf{e}_{x'} + 1\mathbf{e}_{y'} = \frac{1}{\sqrt{2}} \left( \mathbf{e}_x + \mathbf{e}_y \right) = \mathbf{e}_{y'}`

We can see that there are two ways to represent point :math:`\mathbf{p}_3`, either in ``basis1`` or ``basis2``.
In BTensor, we do not need to worry about ``point1`` and ``point2`` being defined in different bases---as
long as the bases are **compatible** (i.e., the belong to the same basis tree), numerical operations, such as addition
can be carried out. For example

.. code-block:: python

    point3 = point1 + point2
    print(f"point3 in basis1: {point3.to_numpy(basis=basis1)}")
    print(f"point3 in basis2: {point3.to_numpy(basis=basis2)}")

returns

.. code-block:: console

    point3 in basis1: [0.70710678 0.70710678]
    point3 in basis2: [0. 1.]

which agrees with the above result.

Basis Transformation
--------------------

While the aim of BTensor is to implement abstract tensor types, which can be operate within...


Rotatation versus Permutation
-----------------------------

In the example above, the derived basis ``basis2`` was defined via the :math:`2 \times 2` transformation matrix.
In general, any derived basis can be defined in terms of a :math:`m \times n` matrix, where :math:`m` is the size
of the parent basis, :math:`n` the size of the derived basis, and :math:`0 < n \leq m`.
If :math:`n = m`, the parent and derived basis span the same space and we consider the derived basis to be a **rotation**
of its parent basis. If however, :math:`n < m`, then the derived basis only spans a **subspace** of its parent basis,
which we can think of as a **rotation + projection** operation.

.. note::
    If parent or derived basis are non-orthogonal, their transformation matrix will not generally be a rotation matrix in the mathematical sense (orthogonal matrix with determinant 1).

Often, we are dealing with derived bases which derive from their parent basis in a simpler way.
For example, we might be interested in the derived basis defined by the first two out of four basis vectors of its parent basis.
While this transformation can be represented in terms of the matrix

.. math::
    \begin{bmatrix}
    1 & 0  \\
    0 & 1  \\
    0 & 0  \\
    0 & 0  \\
    \end{bmatrix}

we can represent it easier in terms of a **indexing array**, a **slice**, or a **masking array**:

- **Indexing array**: a 1D array of integer indices, which refer to the basis vectors of the parent basis. In this example: ``[0, 1]``.
- **Slice**: a slice object with start, stop, and step attributes. In this example: ``slice(0, 2, 1)`` (or simply ``slice(2)``).
- **Masking array**: a 1D array with boolean values, indicating if the corresponding basis vector of the parent basis is included in the derived basis. In this example: ``[True, True, False, False]``.

Opposed to the more general rotation above, we refer to these relations as **permutations**, since indexing array can change the order of basis vectors
(or **permutation + selection**, if the derived basis is smaller than its parent).
Defining a derived basis via a permutation is not purely for convenience, transformation can also be carried out more efficiently in this case.


Multidimensional Tensors
------------------------

In the examples above, we only considered a 1-D vector, with a single associated basis.
How can we work with higher-dimensional tensors? We simply have to use tuples of ``Basis`` instances, i.e.

.. literalinclude:: ../../examples/03-2d-tensor.py
    :linenos:
    :lines: 15-

Note that ``basis2[1]`` with size 2 only spans a subspace of ``basis1[1]`` with size 3.
As a result, ``tensor1`` and ``tensor2`` are created using NumPy arrays of different shapes,
:math:`2 \times 3` and :math:`2 \times 2`, respectively.
While it would not be possible to add the NumPy arrays directly, we can add the corresponding ``Tensor`` objects, since their bases are compatible along each dimension.

The resulting ``tensor3`` can be transformed to any other basis


converted back to its array representation with respect to ``basis1``, as shown in line 15.
However, if we tried the do the same using ``basis2`` and without the additional keyword ``project=True``,
an exception would occur. The reason for this is, that ``basis2`` cannot represent this tensor without loss of information.

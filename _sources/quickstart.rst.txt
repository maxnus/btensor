.. include:: links.rst

==========
Quickstart
==========

Defining Bases and Tensors
--------------------------

As a first example, consider the 2D euclidian basis with basis vectors :math:`\mathbf{e}_x` and :math:`\mathbf{e}_y`,
and a second (non-orthogonal) basis, with basis vectors
:math:`\mathbf{e}_{x'}` =  :math:`\mathbf{e}_x` and
:math:`\mathbf{e}_{y'}` =  :math:`\frac{1}{\sqrt{2}} \left( \mathbf{e}_x + \mathbf{e}_y \right)`.
In other words, the first basis vector is identical, however the second basis vector is rotated 45° clockwise.

A :ref:`Basis <api/_autosummary/btensor.basis.Basis:Basis>` can be defined according to

.. code-block:: python

    import numpy as np
    from btensor import Basis

    basis1 = Basis(2)
    r = np.asarray([[1, 1/np.sqrt(2)],
                    [0, 1/np.sqrt(2)]])
    basis2 = Basis(r, parent=basis1)

where ``basis1`` represents the euclidian 2D basis, ``r`` the transformation matrix, and ``basis2`` the second,
non-orthogonal basis.
The definition of ``basis1`` is very simple: only an integer defining the dimensionality of the space is
required.
In contrast, ``basis2`` is defined in terms of a transformation matrix and a parent basis, namely ``basis1``.
Note that the :ref:`make_subbasis <api/_autosummary/btensor.basis.Basis.make_subbasis:Basis.make\\_subbasis>` method of
``basis1`` could have been used instead.

In BTensor, bases are organized in a **tree structure**. We distinguish two types of bases:

- A **root-basis** does not have a parent and is constructed from an integer size argument.
- A **derived basis** has a parent basis and is defined in terms of a transformation wrt to its parent.

In this example, ``basis1`` is a root-basis and ``basis2`` is a derived basis.

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
differents bases. In particular, it does not make sense to add these representations directly, but instead we have to
consider the basis vectors they represent, such that
:math:`\mathbf{p}_3 = \mathbf{p}_1 + \mathbf{p}_2 = -1\mathbf{e}_{x} + 1\mathbf{e}_{x'} + 1\mathbf{e}_{y'} =
\frac{1}{\sqrt{2}} \left( \mathbf{e}_x + \mathbf{e}_y \right) = \mathbf{e}_{y'}`

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

Basis from Permutation
----------------------

In the example above the derived basis ``basis2`` was defined in terms of ``basis1`` via the :math:`2 \times 2`
transformation matrix.
In general, any derived basis can be defined in terms of a :math:`m \times n` matrix, where :math:`m` is the size
of the parent basis, :math:`n` the size of the derived basis, with :math:`0 < n \leq m`.
If :math:`n = m`, the parent and derived basis span the same space and we consider the derived basis to be a
**rotation** [#f1]_ of its parent basis. If however, :math:`n < m`, then the derived basis only spans a **subspace** of
its parent basis,
which we can think of as a **rotation + projection** operation.

Often, we are dealing with derived bases which derive from their parent basis in a simpler way.
For example, we might be interested in the derived basis defined by the first two out of four basis vectors of its
parent basis.
While this transformation can be represented in terms of the matrix

.. math::
    \begin{bmatrix}
    1 & 0  \\
    0 & 1  \\
    0 & 0  \\
    0 & 0  \\
    \end{bmatrix}

we can represent it easier in terms of a **indexing array**, a **slice**, or a **masking array**:

- **Indexing array**: a 1D array of integer indices, which refer to the basis vectors of the parent basis.
  In this example: ``[0, 1]``.
- **Slice**: a slice object with start, stop, and step attributes. In this example: ``slice(0, 2, 1)``
  (or simply ``slice(2)``).
- **Masking array**: a 1D array with boolean values, indicating if the corresponding basis vector of the parent basis
  is included in the derived basis. In this example: ``[True, True, False, False]``.

In contrast to to the more general rotation above, we refer to these relations as **permutations**, since indexing array can change the order of basis vectors
(or **permutation + selection**, if the derived basis is smaller than its parent).
Defining a derived basis via a permutation when possible is not only more convention, it will also allow for more
efficient transformations between different bases.

.. rubric:: Footnotes

.. [#f1] If parent or derived basis are non-orthogonal, their transformation matrix will not generally be a rotation
         matrix in the mathematical sense (orthogonal matrix with determinant 1).

Active and Passive Transformations
----------------------------------

The current basis of a tensor can be accessed via the ``basis``-attribute:

.. code-block:: python

   >>> print(point3.basis)
   (Basis(id= 1, size= 2, name= Basis1),)

Note that the basis is stored as a tuple, to support multidimensional tensors (see section below).
To change the basis of a tensor, the ``[]``-operator can be used:

.. code-block:: python

   >>> print(point3[basis2].basis)
   (Basis(id= 2, size= 2, name= Basis2),)

When changing the basis using the ``[]``-operator, the ndarray representation will be updated automatically:

.. code-block:: python

   >>> print(point3.to_numpy())
   [0.70710678 0.70710678]
   >>> print(point3[basis2].to_numpy())
   [0. 1.]

This is an example of a **passive** transformation, meaning that while basis and representation change, the (abstract)
point itself does not move in space.

On the other hand, the
:ref:`replace_basis <api/_autosummary/btensor.tensor.Tensor.replace_basis:Tensor.replace\\_basis>` method can be used
to replace the basis while keeping the representation fixed:

.. code-block:: python

   >>> point4 = point3.replace_basis(basis2)
   >>> print(point4.basis)
   (Basis(id= 2, size= 2, name= Basis2),)
   >>> print(point4.to_numpy())
   [0.70710678 0.70710678]

The ``replace_basis`` method can only be used with a basis, that has exactly the same size as the current basis of the
tensor (otherwise it would be impossible to reinterpret the existing representation as referring to the new basis).
For multidimensional tensors, this requirement needs to hold for each dimensions individually.

Changing the basis using ``replace_basis`` is an **active** transformation and consequently ``point4`` describes a
different point in space than ``point3``.


Projection and Spaces
---------------------

When using the ``[]``-operator to perform a change of basis, it is possible to use a basis which is not large enough to
to describe the tensor fully:

.. code-block:: python

   >>> basis3 = Basis([0], parent=basis1)
   >>> print(point3[basis3].to_numpy())
   [0.70710678]

In this case, the ``[]``-operator does not only perform a change-of-basis operations, it also performs a **projection**
onto the subspace spanned by ``basis3``.
In this process, information about the original tensor will be lost and cannot be restored, even when
transforming back into the original basis:

.. code-block:: python

    >>> print(point3.to_numpy())
   [0.70710678 0.70710678]
    >>> print(point3[basis3][basis1].to_numpy())
   [0.70710678 0.        ]

.. note::

   To make sure that the information is lost when performing a change of basis, the
   :ref:`change_basis <api/_autosummary/btensor.tensor.Tensor.change_basis:Tensor.change\\_basis>` method
   or the ``cob``-interface can be used.
   In this way, a ``BasisError`` exception will be raised, when trying to perform a transformation
   which would lead to a loss of information:

   .. code-block:: python

      >>> print(point3.cob[basis3].to_numpy())
      Traceback (most recent call last):
      ...
      btensor.exceptions.BasisError: (Basis(id= 3, size= 1, name= Basis3),) does not span (Basis(id= 1, size= 2, name= Basis1),)


To check if two bases span the same space or are in a sub- or super-space relationship to each other,
the ``space``-attribute can be used in combination with the usual comparison operators:

   .. code-block:: python

      >>> print(basis1.space == basis2.space)
      True
      >>> print(basis1.space == basis3.space)
      False
      >>> print(basis3.space < basis1.space)
      True

Furthermore, the ``|``-operator can be used to check if two bases are orthogonal:

   .. code-block:: python

      >>> basis4 = Basis([1], parent=basis1)
      >>> print(basis3.space | basis4.space)
      True

Multidimensional Tensors
------------------------

So far we have only considered a 1D tensor, a vector, with a single associated basis.
How can we work with higher-dimensional tensors in BTensor? We simply have to use tuples of ``Basis`` instances, i.e.

.. literalinclude:: ../../examples/03-2d-tensor.py
    :linenos:
    :lines: 15-

Note that ``basis2[1]`` with size 2 only spans a subspace of ``basis1[1]`` with size 3.
As a result, ``tensor1`` and ``tensor2`` are created using ndarrays of different shapes,
:math:`2 \times 3` and :math:`2 \times 2`, respectively.
While it would not be possible to add the NumPy arrays directly, we can add the corresponding ``Tensor`` objects,
since their bases are compatible along each dimension.
The resulting ``tensor3`` can be transformed to both ``basis1`` or ``basis2``, as shown in lines 15, 16.

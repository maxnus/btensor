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

In BTensor_, these bases can be defined according to

.. code-block:: python

    import numpy as np
    from btensor import Basis

    basis1 = Basis(2)
    tfm = np.asarray([[1, 1/np.sqrt(2)],
                      [0, 1/np.sqrt(2)]])
    basis2 = Basis(tfm, parent=basis1)

where ``basis1`` represents the euclidian 2D basis, ``tfm`` the transformation matrix, and ``basis2`` the second, non-orthogonal basis.
Note that the definition of ``basis1`` is very simple: only an integer defining the dimensionality of the basis is required.
In contrast, ``basis2`` is defined in terms of a transformation matrix and a parent basis, namely ``basis1``.

In BTensor, bases are organized in a **tree structure**. We distinguish two types of bases:

- A **root basis** does not have a parent and is constructed from an integer size argument.
- A **derived basis** has a parent basis and is defined in terms of a transformation wrt to its parent.

In this example, ``basis1`` is a root basis and ``basis2`` is a derived basis.

.. note::

    The root basis is not required to be orthogonal. For more details on non-orthogonality see TODO

All bases which belong to the same basis tree are considered **compatible**, i.e., BTensor can perform numerical
operations such as addition between tensors expressed in these bases.


For this we require the second fundamental type, the Tensor, which wrapps NumPy_'s ndarray.
Let us consider the points :math:`\mathbf{p}_1 = -1\mathbf{e}_{x} + 1\mathbf{e}_{y}` and :math:`\mathbf{p}_2 = 1\mathbf{e}_{x'} + 1\mathbf{e}_{y'}`.
We can construct these as follows:

.. code-block:: python

    from btensor import Tensor
    point1 = Tensor([-1, 0], basis=basis1)
    point2 = Tensor([ 1, 1], basis=basis2)

The important thing to note is that the representations :math:`(-1, 0)` and :math:`(1, 1)` of these two points refer to
differents bases. In particular, it does not make sense to add these row vectors directly, but instead we have to consider the basis vectors they represent, such that
:math:`\mathbf{p}_3 = \mathbf{p}_1 + \mathbf{p}_2 = -1\mathbf{e}_{x} + 1\mathbf{e}_{x'} + 1\mathbf{e}_{y'} = \frac{1}{\sqrt{2}} \left( \mathbf{e}_x + \mathbf{e}_y \right) = \mathbf{e}_{y'}`

We can see that there are two ways to represent point :math:`\mathbf{p}_3`, either in basis 1 or basis 2.
In BTensor, we do not need to worry about ``point2`` and ``point2`` being defined in different bases---as
long as the bases are **compatible** (i.e., the belong to the same basis tree), numerical operations, such as addition
can be carried out. For example

.. code-block:: python

    point3 = point1 + point2
    print(f"Point 3 in basis 1: {point3.to_array(basis=basis1)}")
    print(f"Point 3 in basis 2: {point3.to_array(basis=basis2)}")

returns

.. code-block:: console

    Point 3 in basis 1: [0.70710678 0.70710678]
    Point 3 in basis 2: [0. 1.]

which agrees with the above result.

Multidimensional Tensors
------------------------

In the example above, we only considered a 1D tensor, with a single associated basis.
How can we work with higherdimensional tensors? We simply have to work with tuples of ``Basis`` instances, i.e.

.. literalinclude:: ../examples/02-matrix.py
    :linenos:

In this example, both ``basis1`` and ``basis2`` are tuples of ``Basis`` instances.
We also see a new way of defining derived bases in lines 6 and 7, via a one-dimensional **indexing array**;
such an array do
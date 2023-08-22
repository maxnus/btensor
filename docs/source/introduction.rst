.. include:: links.rst

.. _introduction:

============
Introduction
============


In some formalisms, such as the `Bra-ket notation <https://en.wikipedia.org/wiki/Bra-ket_notation>`_
used in quantum mechanics, one distinguishes between the vector
:math:`\mathbf{v}` as an abstract, mathematical object
and a representation of that vector in some basis, :math:`v`.
Specifically, :math:`\mathbf{v}` can be represented in terms of a basis :math:`\mathbf{b}_i`
and its components :math:`v^i` in that basis as

.. math::
    :label: eq_1

    \mathbf{v} = \sum_i v^i \: \mathbf{b}_i.

..
    where `Einstein notation <https://en.wikipedia.org/wiki/Einstein_notation>`_ was used.

In numerical calculations we exclusively deal with the elements :math:`v^i` and usually call this array of numbers
the vector itself.
However, care needs to be taken when working with these arrays of numbers; for example when adding two vector
representations according to :math:`v^i = u^i + w^i`, all representations must refer to the same basis in order
for the addition to be meaningful. Further restrictions apply when dealing with non-orthogonal bases [#f1]_.
This can be contrasted with the completely general expression :math:`\mathbf{v} =  \mathbf{u} + \mathbf{w}`,
which is considered true without making reference to any basis.

When dealing with vectors, or higher dimensional analogues, here called **tensors**, in different bases,
being able to work with their abstract versions can be a big advantage, for a number of reasons:

#. It avoids the mental overhead of having to keep track of which representation corresponds to which basis.
   In fact, without abstract tensor types, the basis often has to be documented alongside the code itself, either
   via a suitable variable name, or in a docstring. It is advantageous to attach this knowledge to the tensor
   type itself.
#. It prevents illegal operations, such as adding two representation corresponding to different bases. Such a mistake
   would otherwise only become apparent if it results in a shape missmatch (which is only the case if the bases have
   different sizes).
#. It saves the developer from having to repeatedly implement change of basis transformations, which takes time
   and can be a cause of errors.

BTensor was developed to allow working with such abstract tensors and taking benefit from the listed advantages.

A Tensor with Basis
-------------------

Equation :eq:`eq_1` tells us what is needed to represent an
abstract vector or tensor: a representation, i.e. an array of numbers, and a basis.
However, a closer look at this equation reveals that each :math:`\mathbf{b}_i` in itself is an abstract vector
(note the bold font).
We can again assume that these abstract vectors can be represented by some numbers :math:`r_i^{\hphantom{i}j}`
(now forming a matrix) and yet another basis, :math:`\mathbf{c}_j`, such that we arrive at

.. math::

    \mathbf{v} = \sum_i v^i \sum_j r_i^{\hphantom{i}j} \: \mathbf{c}_j.

Repeating the same step again, we would obtain

.. math::

    \mathbf{v} = \sum_i v^i \sum_j r_i^{\hphantom{i}j} \sum_k s_j^{\hphantom{j}k} \: \mathbf{d}_k,

where we introduced yet another basis and matrix of numbers, :math:`\mathbf{d}_k` and :math:`s_j^{\hphantom{j}k}`,
but no matter how often we do this, there will always be another
set of abstract basis vectors on the very right-hand side.
We thus have to terminate this process at some basis (called *root basis*) and do what we have been
doing all along when working with vector representations: we have to accept that the basis of the root basis itself
remains *implicit*, in other words only exist in our head (and hopefully as a suitably chosen variable name),
without any analogue in BTensor [#f2]_.

Once is done, all other bases can be represented in terms of the root basis, following the chain of
matrix products above.
The matrix elements which encode the basis transformations, :math:`r_i^{\hphantom{i}j}`,
are the inner products

.. math::

    r_i^{\hphantom{i}j} = \langle \mathbf{c}^j , \mathbf{b}_i \rangle,

and relate one basis to the next. They are the input required to define a new basis, based on an existing one
(starting from the root basis).  Note that the bases do not have to form a linear chain of dependency, as shown here.
There can be mupliple bases deriving from the same parent basis, forming a tree structure as a result.

Once all the desired bases have been defined in this way, we can start creating abstract tensors:
the required input is a set of bases (one for each tensor dimension) and the representation of the tensor within
this set of bases.
Any tensors defined in this way can be added or contracted with one another, even if they do not have the same
set of bases: the only requirement is that all bases derive from the same root basis, such that the basis transformation
can be resolved automatically.

.. rubric:: Footnotes

.. [#f1] In a non orthogonal basis, vector representations can only be added if they both transform in the same way:
         covariantly or contravariantly. In contrast, a contraction over one dimension is only allowed if the
         two representations transform differently in that dimension.

.. [#f2] In the case of a non-orthogonal root basis it is also necessary to define the metric
         :math:`M_{ij} = \langle \mathbf{b}_i , \mathbf{b}_j \rangle`
         of this basis, which is used to change the transformation behavior between co- and contravariant
         whenever necessary. For an orthonormal root basis the metric is simply the identity matrix.

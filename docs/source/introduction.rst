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

In numerical calculations we exclusively deal with the elements :math:`v^i` and often call this
the vector itself.


is used instead of the geometric object
:math:`\mathbf{v}` itself.
For example, in a vector addition such as :math:`\mathbf{v} =  \mathbf{u} + \mathbf{w}` no reference is made to the basis
in which the addition is made; instead this identity is more fundamental.
In contrast, the addition of the respective representations, :math:`v^i = u^i + w^i`, is only meaningful if all
representations refer to the same underlying basis.

In some frameworks, such as `quantum embedding methods
<https://journals.aps.org/prx/abstract/10.1103/PhysRevX.12.011046>`_ one typically deals with a large number of
different bases, such that frequent transformation between different representations are required, to perform
operations such as additions or tensor contractions.
To this end, it would be advantageous to store and modify the abstract vector

BTensor_ was developped to




When performing numerical calculations, we deal with representations, such as :math:`v^i`,
of the vector, not the mathematical vector :math:`\mathbf{v}` itself.


More
----

.. math::
    \langle \mathbf{b}^i , \mathbf{c}_j \rangle = R^i_j

wad





More generally,
a tensor :math:`\mathbf{T}` with rank :math:`n \ge 2` can be represented in terms of a set of bases,
:math:`\mathbf{b}, \mathbf{b}', \dots`, i.e., :eq:`eq_1`


.. math::

    \mathbf{T} = \sum_{ij\dots} t^{ij\dots} \: \mathbf{b}_{i} \: \mathbf{c}_{j} \dots.


When working vectors ors tensors in a program, we generally just store and modify the array of numbers :math:`v^i`
or :math:`t^{ij\dots}`, whereas the knowledge of the underlying basis remains implicit.

Often, function just work with a tensor representation in a specific basis, and this basis has to be known by the user
or developer. Similarly, a function might return a tensor representation in a specific basis. The documentation of
which choice of basis is acceptable has to happen alongside the code itself, either via a suitable function name or
docstring. If such functions are used with a tensor representation in the wrong type of basis, the


An array with basis
-------------------

BTensor_ addresses
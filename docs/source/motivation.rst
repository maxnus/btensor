.. include:: links.rst

.. _motivation:

==========
Motivation
==========


A vector :math:`\mathbf{v}` can be represented in terms of a basis :math:`\mathbf{b}_i`
and its components in that basis, :math:`v^i`, i.e.,

.. math::

    \mathbf{v} = v^i \: \mathbf{b}_i,

where `Einstein notation <https://en.wikipedia.org/wiki/Einstein_notation>`_ was used. More generally,
a tensor :math:`\mathbf{T}` with rank :math:`n \ge 2` can be represented in terms of a set of bases,
:math:`\mathbf{b}, \mathbf{b}', \dots`, i.e.,


.. math::

    \mathbf{T} = t^{ij\dots} \: \mathbf{b}_{i} \: \mathbf{b}_{j}' \dots.


When working vectors ors tensors in a program, we generally just store and modify the array of numbers :math:`v^i`
or :math:`t^{ij\dots}`, whereas the knowledge of the underlying basis remains implicit.

Often, function just work with a tensor representation in a specific basis, and this basis has to be known by the user
or developer. Similarly, a function might return a tensor representation in a specific basis. The documentation of
which choice of basis is acceptable has to happen alongside the code itself, either via a suitable function name or
docstring. If such functions are used with a tensor representation in the wrong type of basis, the


An array with basis
-------------------

BTensor_ addresses
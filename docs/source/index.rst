.. BTensor documentation master file, created by
   sphinx-quickstart on Mon May  8 10:23:03 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. include:: links.rst

BTensor Documentation
=====================

BTensor_ defines the Tensor type, a generalization of NumPy_'s ndarray, which can store a basis along each dimension.
When performing tensor operations, such as additions or contractions, BTensor will ensure that:

- The bases of the tensors are compatible along all dimensions or raise an exception if not
- If the bases are compatible, but not equal, necessary basis transformations will be carried out automatically


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   quickstart
   api/index


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

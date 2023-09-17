# BTensor

BTensor defines a tensor type, a generalization of NumPy's ndarray, which can store a basis along each dimension.
When performing tensor operations, such as additions or contractions, BTensor will ensure that:

- The bases of the tensors are compatible along all dimensions or raise an Exception if not
- If the bases are compatible, but not equal, necessary basis transformations will be carried out automatically

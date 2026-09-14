[![CI](https://github.com/maxnus/btensor/actions/workflows/ci.yml/badge.svg)](https://github.com/maxnus/btensor/actions/workflows/ci.yml)
[![Sphinx: Render docs](https://github.com/maxnus/btensor/actions/workflows/sphinx.yml/badge.svg)](https://github.com/maxnus/btensor/actions/workflows/sphinx.yml)

BTensor
=======

BTensor defines a tensor type, a generalization of NumPy's ndarray, which can store a basis along each dimension.
When performing tensor operations, such as additions or contractions, BTensor will ensure that:

- The bases of the tensors are compatible along all dimensions or raise an exception if not
- If the bases are compatible, but not equal, necessary basis transformations will be carried out automatically

Please refer to the [documentation](https://maxnus.github.io/btensor/index.html) for more information.

Development
-----------

This project uses [uv](https://docs.astral.sh/uv/) for dependency management
and [ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
uv sync --all-groups            # create the environment
uv run pytest tests             # run the test suite
uv run pytest -n auto tests     # ... in parallel
uv run ruff check .             # lint
uv run ruff format .            # format
```

To run the suite against a different interpreter, pass it to uv directly:

```bash
uv run --python 3.14 pytest tests
```

Build the documentation with:

```bash
uv run --group docs sphinx-build -W -b html docs/source docs/build/html
```

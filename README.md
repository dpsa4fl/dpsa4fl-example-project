
# dpsa4fl example project (python & flower)

This is an example for how to do differentially private federated machine learning
with the [flower](https://flower.dev/) framework, using the [dpsa4fl](https://github.com/dpsa-project/dpsa4fl)
library to interact with aggregation servers as described by this [internet draft](https://github.com/ietf-wg-ppm/draft-ietf-ppm-dap).

## Development

This is a python project that uses the [poetry](https://python-poetry.org/) package manager, which means
that you need to have `poetry` installed on your machine.

To start developing, you need to enable a virtual environment. This can be done easily with poetry:
```fish
path/dpsa4fl-example-project> poetry shell
```
Now all dependencies (including `dpsa4fl-bindings`) can be installed by simply saying
```fish
(dpsa4fl-example-project-py3.10) path/dpsa4fl-example-project> poetry install
```

## Python package management tips
1. If poetry hangs, then maybe [this](https://github.com/python-poetry/poetry/issues/6906#issuecomment-1298972506) will help.



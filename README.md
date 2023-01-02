
# dpsa4fl example project (via dpsa4fl-bindings.py & flower)

This is an example for how to do differentially private federated machine learning
with the [flower](https://flower.dev/) framework, using the [dpsa4fl](https://github.com/dpsa-project/dpsa4fl)
library to interact with aggregation servers as described by this [internet draft](https://github.com/ietf-wg-ppm/draft-ietf-ppm-dap).

The project itself is based on the [pytorch quickstart example](https://github.com/adap/flower/tree/main/examples/quickstart_pytorch)
from flower (learning the CIFAR-10 dataset), but is adapted to use our [flower fork](https://github.com/dpsa-project/flower).

**Note: Currently no differential privacy is employed. Only secure aggregation via janus.**

## First time setup
Running this project consists of four steps:

### 1. Installing dependencies
You need to have the [poetry](https://python-poetry.org/) package manager and [docker](https://www.docker.com/)
(with [docker-compose](https://docs.docker.com/compose/)) installed on your system. See the links for appropriate installation instructions
for your system.

### 2. Setting up a local janus instance

Clone the [dpsa4fl-testing-infrastructure](https://github.com/dpsa-project/dpsa4fl-testing-infrastructure)
repository into a directory of your choice (here `~`).
```fish
~> git clone https://github.com/dpsa-project/dpsa4fl-testing-infrastructure
```
Switch into that directory and into the `run` subfolder.
```fish
~> cd dpsa4fl-testing-infrastructure/run
```
Start the containers with docker-compose.
```fish
~/dpsa4fl-testing-infrastructure/build> docker-compose up -d
```

### 3. Get and prepare this example project

Clone this repo into a directory of your choice (here again `~`).
```fish
~> git clone https://github.com/dpsa-project/dpsa4fl-example-project
```
Switch into this directory.
```fish
~> cd dpsa4fl-example-project
```
Enable a python virtual environment using poetry.
```fish
~/dpsa4fl-example-project> poetry shell
```
Install dependencies with poetry.
```fish
(dpsa4fl-example-project-py3.10) ~/dpsa4fl-example-project> poetry install
```

### 4. Run flower server and two clients
For this, it is best if you open individual shells for each instance of server and clients.

So in the current shell, start the flower server.
```fish
(dpsa4fl-example-project-py3.10) ~/dpsa4fl-example-project> python dpsa4fl_example_project/server.py
```

Also, do the following two times:
 - Open new shell.
 - Change into `~/dpsa4fl-example-project` directory (use your path).
 - Enable a virtual environment like above with `poetry shell`.
 - Run the flower client with `python dpsa4fl_example_project/client.py`.
 
## Result
When the server is running, and the two clients are connected, the learning procedure will start automatically.
The server will let the clients train multiple times on the dataset, interspersed with evaluation rounds. The
gradients of each training round are aggregated via the local janus server instance.

In the end, the accuracy of each evaluation round is printed. You will see output similar to the following.
```
INFO flower 2022-12-28 12:24:36,360 | server.py:144 | FL finished in 997.9534725520025
INFO flower 2022-12-28 12:24:36,360 | app.py:192 | app_fit: losses_distributed [(1, 2.058671712875366), (2, 1.5741422176361084), (3, 1.2739187479019165), (4, 1.1578938961029053), (5, 1.0813101530075073), (6, 1.0532516241073608), (7, 1.016814112663269)]
INFO flower 2022-12-28 12:24:36,360 | app.py:193 | app_fit: metrics_distributed {'accuracy': [(1, 0.3825), (2, 0.4958), (3, 0.5599), (4, 0.5891), (5, 0.6189), (6, 0.6333), (7, 0.6403)]}
```
After the seventh round, the accuracy is at 64%, which is comparable to the accuracy of the original flower example without secure aggregation.

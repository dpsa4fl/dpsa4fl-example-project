import warnings
from collections import OrderedDict

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

from flwr.common.typing import Config, NDArrays, NDArray, Scalar
import numpy as np

from dpsa4fl_bindings import client_api__new_state, client_api__submit


# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def train(net, trainloader, epochs):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for images, labels in tqdm(trainloader):
            optimizer.zero_grad()
            criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()
            optimizer.step()


def test(net, testloader):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            outputs = net(images.to(DEVICE))
            labels = labels.to(DEVICE)
            loss += criterion(outputs, labels).item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    return loss / len(testloader.dataset), correct / total


def load_data():
    """Load CIFAR-10 (training and test set)."""
    trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = CIFAR10("./data", train=True, download=True, transform=trf)
    testset = CIFAR10("./data", train=False, download=True, transform=trf)
    return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Load model and data (simple CNN, CIFAR-10)
net = Net().to(DEVICE)
trainloader, testloader = load_data()

# Define Flower client
class FlowerClient(fl.client.NumPyClient):

    ######################################################################
    # BEGIN
    def reshape_parameters(self, parameters: NDArrays) -> NDArrays:
        # update parameter shapes
        # if we are in first round (self.shapes is None), then we don't need to reshape.
        # But if we are in any following round, then we need to take our previous shapes
        # and lengths and reshape the `parameters` argument accordingly
        if (self.shapes is not None) and (self.split_indices is not None):
            assert len(self.split_indices) + 1 == len(self.shapes), "Expected #indices = #shapes - 1"

            print("In follow-up round, reshaping. length of params is: ", len(parameters))
            assert len(parameters) == 1, "Expected parameters to have length 1!"

            single_array = parameters[0]
            print("Found single ndarray of shape ", single_array.shape, " and size ", single_array.size)
            # assert single_array.shape == (,), "Wrong ndarray shape!"

            # split and reshape
            arrays = np.split(single_array, self.split_indices)
            print("After splitting, have ", len(arrays), " arrays")

            arrays = [np.reshape(a,s) for (a,s) in zip(arrays, self.shapes)]
            print("Now have the following shapes:")
            for a in arrays:
                print(a.shape)

            # check that we have our value at position 449
            rval1 = np.around(arrays[0], decimals = 2)
            print("in array at 449 have: ", rval1[0:1, 0:1, 0:1, 0:1])
            print(rval1)

            # change parameters to properly shaped list of arrays
            parameters = arrays

        else:
            print("In first round, not reshaping.")

        return parameters

    def flatten_parameters(self, params: NDArrays) -> NDArray:
        # print param shapes
        print("The shapes are:")
        for p in params:
            print(p.shape)

        # set value at position 1,0,0,0
        # params[0] = np.zeros((6,3,5,5),dtype=np.float32)
        # params[0][1,0,0,0] = 10

        # print("in array at have: ", params[0][0:1, 0:1, 0:1, 0:1])
        # print(params[0])

        # flatten params before submitting
        self.shapes = [p.shape for p in params]
        flat_params = [p.flatten('C') for p in params] #TODO: Check in which order we need to flatten here
        p_lengths = [p.size for p in flat_params]

        # loop
        # (convert p_lengths into indices because ndarray.split takes indices instead of lengths)
        split_indices = []
        current_index = 0
        for l in p_lengths:
            split_indices.append(current_index)
            current_index += l
        split_indices.pop(0) # need to throw away first element of list
        self.split_indices = split_indices


        flat_param_vector = np.concatenate(flat_params)

        # test indices locations
        # flat_param_vector = flat_param_vector - flat_param_vector
        # set position number 449 to 1
        # np.put(flat_param_vector, [449], [0.5])

        print("vector length is: ", flat_param_vector.shape)

        # flat_param_vector = np.zeros((20), dtype=np.float32)
        # print("new vector length is: ", flat_param_vector.shape)

        return flat_param_vector


    # END
    ######################################################################


    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        # print the task_id
        # task_id = config['task_id']
        # print(f"My task_id is: {task_id=}")

        self.set_parameters(parameters)
        train(net, trainloader, epochs=3)

        params = self.get_parameters(config={})

        flat_param_vector = self.flatten_parameters(params)

        norm = np.linalg.norm(flat_param_vector)
        print("norm of vector is: ", norm)
        # if norm > 2:
        #     print("Need to scale vector")
        #     flat_param_vector = flat_param_vector * (2/(norm + 0.01))
        #     norm = np.linalg.norm(flat_param_vector)
        #     print("now norm of vector is: ", norm)

        params = self.reshape_parameters([flat_param_vector])

        return params, len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}

dpsa4fl_client_state = client_api__new_state(62006)

# Start Flower client
fl.client.start_numpy_client(
    server_address="127.0.0.1:8081",
    client=fl.client.DPSANumPyClient(dpsa4fl_client_state, FlowerClient()),
)

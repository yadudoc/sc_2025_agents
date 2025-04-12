import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import copy
import numpy as np
import networkx as nx
import intel_extension_for_pytorch as ipex

def calculate_model_size(model: nn.Module) -> float:
    """ From https://stackoverflow.com/questions/71851474/how-to-find-the-size-of-a-deep-learning-model
    """
    param_size = 0
    buffer_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb


class CNN(nn.Module):
    def __init__(self, model_size:str="small"):
        super(CNN, self).__init__()
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.model_size = model_size

        # Note: These model size are tweaked to get match previous
        # model sizes used for experiments and do not match realistic
        # parameters for real use-cases
        if self.model_size == "small":
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.fc1 = nn.Linear(9216, 32)
            self.fc2 = nn.Linear(32, 10)

        elif self.model_size == "medium":
            self.conv1 = nn.Conv2d(1, 64, 3, 1)
            self.conv2 = nn.Conv2d(64, 64, 3, 1)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 320)

        elif self.model_size == "large":
            self.conv1 = nn.Conv2d(1, 128, 3, 1)
            self.conv2 = nn.Conv2d(128, 64, 3, 1)
            self.fc1 = nn.Linear(9216, 256)
            self.fc2 = nn.Linear(256, 464)

        elif self.model_size == "largex2":
            self.conv1 = nn.Conv2d(1, 256, 3, 1)
            self.conv2 = nn.Conv2d(256, 64, 3, 1)
            self.fc1 = nn.Linear(9216, 512)
            self.fc2 = nn.Linear(512, 308)

        else:
            raise ValueError(f"Model size: {model_size} is not supported")

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class Node:
    def __init__(self, node_id, model, train_data, device):
        self.node_id = node_id
        self.model = copy.deepcopy(model)
        self.train_data = train_data
        self.device = device
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

    def train_local(self, epochs=1):
        self.model.train()

        for epoch in range(epochs):
            for data, target in self.train_data:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                self.optimizer.step()

    def average_with_neighbors(self, neighbor_state_dicts):
        # Average model parameters with neighboring nodes
        averaged_state_dict = copy.deepcopy(self.model.state_dict())
        neighbor_count = 1  # Include self in count

        # Collect and average parameters from neighbors
        for neighbor_state_dict in neighbor_state_dicts:
            for key in averaged_state_dict.keys():
                averaged_state_dict[key] += neighbor_state_dict[key]
            neighbor_count += 1

        # Compute average
        for key in averaged_state_dict.keys():
            averaged_state_dict[key] = torch.div(averaged_state_dict[key], neighbor_count)

        # Update model with averaged parameters
        self.model.load_state_dict(averaged_state_dict)

class DecentralizedLearning:
    def __init__(self, num_nodes, model_size:str='small', rounds:int=1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_nodes = num_nodes
        self.nodes = {}
        self.topology = topology
        self.model_size = model_size
        self.rounds = rounds

    def create_network_topology(self):
        if self.topology == 'ring':
            # Create ring topology
            graph = nx.cycle_graph(self.num_nodes)
        elif self.topology == 'grid':
            # Create grid topology
            size = int(np.sqrt(self.num_nodes))
            grid = nx.grid_2d_graph(size, size)

            # Convert grid coordinates to integers
            mapping = {(x, y): x * size + y for x, y in grid.nodes()}
            graph = nx.relabel_nodes(grid, mapping)
        elif self.topology == 'random':
            # Create random topology
            graph = nx.erdos_renyi_graph(self.num_nodes, 0.3)

        return nx.to_dict_of_lists(graph)

    def initialize_network(self):
        # Load and preprocess MNIST dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        dataset = datasets.MNIST('data', train=True, download=True, transform=transform)

        # Calculate sizes for each node's dataset
        base_size = len(dataset) // self.num_nodes
        remaining = len(dataset) % self.num_nodes
        node_sizes = [base_size + 1 if i < remaining else base_size
                    for i in range(self.num_nodes)]

        # Split dataset for different nodes
        node_data = random_split(dataset, node_sizes)

        # Create dataloaders for each node
        node_loader = DataLoader(dataset, batch_size=64, shuffle=True)

        # Initialize global model architecture (each node will get a copy)
        model = CNN(model_size).to(self.device)

        # Create network topology
        topology = self.create_network_topology()

        # Initialize nodes
        for i in range(self.num_nodes):
            self.nodes[i] = Node(
                node_id=i,
                model=model,
                train_data=node_loaders[i],
                neighbors=topology[i],
                device=self.device
            )

    def train(self, num_rounds, local_epochs):
        for round in range(num_rounds):
            print(f"Round {round + 1}/{num_rounds}")

            # Local training phase
            for node_id, node in self.nodes.items():
                print(f"Training node {node_id + 1}/{self.num_nodes}")
                node.train_local(epochs=local_epochs)

            # Communication phase - each node averages with its neighbors
            for node in self.nodes.values():
                node.average_with_neighbors(self.nodes)

def main():
    # Initialize decentralized learning system
    num_nodes = 9  # Using 9 nodes for a 3x3 grid topology
    dl_system = DecentralizedLearning(num_nodes, topology='grid')
    dl_system.initialize_network()

    # Start training
    num_rounds = 10
    local_epochs = 1
    dl_system.train(num_rounds, local_epochs)

def test_model_sizes():

    for model_size in ["small", "medium", "large", "largex2"]:
        cnn = CNN(model_size=model_size)
        actual_size = calculate_model_size(cnn)
        print(f"{model_size=} {actual_size=}MB")

if __name__ == "__main__":
    test_model_sizes()
    #main()

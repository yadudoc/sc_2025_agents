import numpy as np
from numpy import random
import networkx as nx
import os

np.random.seed(0)
random.seed(0)
"""
In this file we create a few sample network topologies for testing

"""
os.makedirs("topology", exist_ok=True)

topology = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
np.savetxt("topology/topo_1.txt", topology, fmt="%d")

topology = np.array([[0, 1, 0, 1], [1, 0, 1, 1], [0, 1, 0, 0], [1, 1, 0, 0]])
np.savetxt("topology/topo_2.txt", topology, fmt="%d")

topology = np.array(
    [
        [0, 1, 0, 0, 0, 0],
        [1, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1],
        [0, 0, 1, 0, 1, 0],
    ]
)
np.savetxt("topology/topo_3.txt", topology, fmt="%d")

topology = np.array(
    [
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
    ]
)
np.savetxt("topology/topo_4.txt", topology, fmt="%d")

topology = np.array(
    [
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
    ]
)
np.savetxt("topology/topo_24.txt", topology, fmt="%d")


G = nx.powerlaw_cluster_graph(24, 2, p=0.01, seed=0)
# randomly assign edge weights (network connection probabilities)
for u, v, w in G.edges(data=True):
    # weight with chance of sustaining network connection
    w["weight"] = random.choice([0.7, 0.8, 0.9, 1], p=[0.03, 0.07, 0.3, 0.6])


topology = nx.to_numpy_array(G)
np.savetxt("topology/topo_24.txt", topology, fmt="%d")
print(topology)

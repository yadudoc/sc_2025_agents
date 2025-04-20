from __future__ import annotations

import aeris
import random
import numpy as np
import argparse
import logging
import time
import os
from datetime import datetime

# from typing import Self
import threading
from concurrent.futures import Future

from aeris.agent import Agent
from aeris.behavior import Behavior
from aeris.behavior import loop
from aeris.behavior import action
from aeris.handle import Handle
from aeris.handle import HandleDict
from aeris.handle import HandleList
from aeris.handle import ProxyHandle
from aeris.handle import UnboundRemoteHandle
from aeris.exchange.thread import ThreadExchange
from aeris.exchange.redis import RedisExchange
from aeris.launcher.executor import ExecutorLauncher
from aeris.launcher.thread import ThreadLauncher
from aeris.logging import init_logging
from proxystore.connectors.endpoint import EndpointConnector
from proxystore.store import Store
from proxystore.store.executor import ProxyAlways
from proxystore.ex.connectors.dim.zmq import ZeroMQConnector

from aeris.exchange.hybrid import HybridExchange
from aeris.exchange.proxystore import ProxyStoreExchange
from aeris.exchange.redis import RedisExchange
from concurrent.futures import ThreadPoolExecutor
from parsl.concurrent import ParslPoolExecutor

## ML bits
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import copy
import numpy as np
import networkx as nx
import intel_extension_for_pytorch as ipex
from decentralized import CNN, Node
from parsl_config import PARSL_CONFIGS
from dml_agent import DMLAgent, Tracker


logger = logging.getLogger(__name__)


def get_adj_dict(topo_file: str) -> dict[list]:

    data = np.loadtxt(topo_file)

    adj_dict: dict[list] = {}

    for row_id, row in enumerate(data):
        neighbors = []
        for col_id, col in enumerate(row):
            if col != 0:
                neighbors.append(col_id)
        adj_dict[row_id] = neighbors

    return adj_dict


def spawn_tracker(exchange, launcher, agent_count):

    node_id = exchange.create_agent()
    node_handle: UnboundRemoteHandle[DMLAgent] = exchange.create_handle(node_id)
    node_behavior = Tracker(agent_count)
    agent_handle = launcher.launch(
        node_behavior,
        exchange,
        agent_id=node_id,
    )
    return node_handle

def spawn_agents(adj_dict: dict[list], exchange, launcher, logpath,
                 tracker_handle: Handle[Tracker]=None,
                 rounds:int=4,
                 model_size:str = "small",
                 epochs_per_round:int=1,
                 ) -> None:

    nodes = {}

    # First create handles for all nodes
    for node in adj_dict:
        node_id = exchange.create_agent()
        node_handle: UnboundRemoteHandle[DMLAgent] = exchange.create_handle(
            node_id)
        #node_handle == exchange.create_handle(node_id)
        nodes[node] = {'node_id' : node_id,
                       'node_handle': node_handle}

    logger.info(f"Launching {len(adj_dict)} nodes")

    # Create behaviors with handles to its neighbors
    for node in nodes:
        neighbor_handles = [nodes[n]['node_handle'] for n in adj_dict[node]]

        node_behavior = DMLAgent(node_id=node,
                                 neighbors=HandleList(neighbor_handles),
                                 tracker=tracker_handle,
                                 logpath=logpath,
                                 rounds=rounds,
                                 model_size=model_size,
                                 epochs_per_round=1,
                                 )

        agent_handle = launcher.launch(
            node_behavior,
            exchange,
            agent_id=nodes[node]['node_id']
        )

        nodes[node]['agent_handle'] = agent_handle

    logger.info("Finished launching agents")

    return [nodes[node]['node_handle'] for node in nodes]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--topo_file", required=True,
                        help="Topology file")
    parser.add_argument("-e", "--environment", default='threads',
                        help="environment threads/parsl")
    parser.add_argument('--log_dir', default="agent_logs",
                        help="log dir")
    parser.add_argument('--redis_hostname',
                        help="redis hostname")
    parser.add_argument('--redis_port', default=6789,
                        help="redis port")
    parser.add_argument('--rounds', default=5,
                        help="Number of rounds to run")
    parser.add_argument('--model_size', default='small',
                        help="Model size: small/medium/large/largex2")

    args = parser.parse_args()

    thread_launcher = ExecutorLauncher(ThreadPoolExecutor(), close_exchange=False)

    print(f"Running in environment:{args.environment}")

    if args.environment == 'threads':
        exchange = ThreadExchange()
        launcher = thread_launcher

    elif args.environment == 'parsl-gpu':
        exchange = RedisExchange(hostname=args.redis_hostname, port=args.redis_port)

        config = PARSL_CONFIGS['htex-aurora-gpu'](run_dir=os.getcwd(), workers_per_node=12)
        executor = ParslPoolExecutor(config)
        launcher = ExecutorLauncher(executor, close_exchange=True)

    elif args.environment == 'parsl-cpu':
        exchange = RedisExchange(hostname=args.redis_hostname, port=args.redis_port)
        config = PARSL_CONFIGS['htex-aurora-cpu'](run_dir=os.getcwd(), workers_per_node=12)
        executor = ParslPoolExecutor(config)
        launcher = ExecutorLauncher(executor, close_exchange=True)

    elif args.environment == 'parsl-gpu-proxy':
        connector = ZeroMQConnector(25780, interface='hsn0', timeout=1)
        store = Store(
            'exchange',
            connector,
            cache_size=0,
            register=True,
        )
        redis_exchange = RedisExchange(hostname=args.redis_hostname, port=args.redis_port)
        exchange = ProxyStoreExchange(
            redis_exchange,
            store,
            should_proxy=ProxyAlways(),
            resolve_async=True, # Modified for experiments
        )

        config = PARSL_CONFIGS['htex-aurora-gpu'](run_dir=os.getcwd(), workers_per_node=12)
        executor = ParslPoolExecutor(config)
        launcher = ExecutorLauncher(executor, close_exchange=True)

    else:
        raise ValueError(f"{args.environment} is not supported")

    if args.log_dir == "agent_logs":
        run_id = len(os.listdir(args.log_dir))
        logpath = f"agent_logs/{run_id:03}"
    else:
        os.makedirs(args.log_dir, exist_ok=True)
        logpath = args.log_dir

    adj_dict = get_adj_dict(args.topo_file)

    agent_count = len(adj_dict)
    print("Agent count: ", agent_count)

    # Tracker agent must be launched with the thread_launcher so as to run
    # on the lead node and not on the limited GPU attached worker slots
    tracker_handle = spawn_tracker(exchange, thread_launcher, agent_count)

    agent_handles = spawn_agents(
        adj_dict, exchange, launcher, logpath,
        tracker_handle=tracker_handle,
        model_size=args.model_size,
        rounds=int(args.rounds),
    )

    thread_launcher.wait(tracker_handle.agent_id)


    for agent_handle in agent_handles:
        agent_client = agent_handle.bind_as_client()
        agent_client.shutdown()
        agent_client.close()

    print("Exiting!!!")

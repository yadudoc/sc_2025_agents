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
from aeris.exchange.hybrid import HybridExchange
from aeris.launcher.executor import ExecutorLauncher
from aeris.launcher.thread import ThreadLauncher
from aeris.logging import init_logging
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
from dml_agent import DMLAgent
from dml_agent import Tracker


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

def spawn_tracker(thread_launcher, exchange) -> Handle:

    node_id = exchange.create_agent()
    print(f"YADU {node_id=}")
    node_handle: UnboundRemoteHandle[DMLAgent] = exchange.create_handle(node_id)

    tracker = Tracker(agent_count=len(adj_dict))
    tracker_handle = thread_launcher.launch(
        tracker,
        exchange,
        agent_id=node_id)

    return tracker_handle


def spawn_agents(adj_dict: dict[list], exchange, launcher, logpath, tracker_handle,
                 model_size: str = 'medium',
                 rounds:int = 5) -> None:

    init_logging(logging.INFO, logfile=f"{logpath}/top_level.log")
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
                                 model_size=model_size,
                                 rounds=rounds)

        agent_handle = launcher.launch(
            node_behavior,
            exchange,
            agent_id=nodes[node]['node_id']
        )

        nodes[node]['agent_handle'] = agent_handle



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--topo_file", required=True,
                        help="Topology file")
    parser.add_argument("-e", "--environment", default='threads',
                        help="environment threads/parsl")
    parser.add_argument("--exchange", default='redis',
                        help="Exchange to use redis/threads/hybrid")
    parser.add_argument('--log_dir', default="agent_logs",
                        help="log dir")
    parser.add_argument('--redis_hostname',
                        help="redis hostname")
    parser.add_argument('--rounds', default=5,
                        help="Number of rounds to run")
    parser.add_argument('--model_size', default='medium',
                        help="Model size (small=1MB, medium=5MB, large=10MB)")
    parser.add_argument('--redis_port', default=6789,
                        help="redis port")
    args = parser.parse_args()


    adj_dict = get_adj_dict(args.topo_file)


    print(f"Running in environment:{args.environment}")
    executor = ThreadPoolExecutor()
    thread_launcher = ExecutorLauncher(executor)

    if args.exchange == 'redis':
        exchange = RedisExchange(hostname=args.redis_hostname, port=args.redis_port)
    elif args.exchange == 'threads':
        exchange = ThreadExchange()
    elif args.exchange == 'hybrid':
        exchange = HybridExchange(redis_host=args.redis_hostname,
                                  redis_port=args.redis_port,
                                  interface='hsn0')

    if args.environment == 'threads':
        launcher = thread_launcher

    elif args.environment == 'parsl-gpu':
        config = PARSL_CONFIGS['htex-aurora-gpu'](run_dir=os.getcwd(), workers_per_node=12)
        executor = ParslPoolExecutor(config)
        launcher = ExecutorLauncher(executor, close_exchange=True)

    elif args.environment == 'parsl-cpu':
        config = PARSL_CONFIGS['htex-aurora-cpu'](run_dir=os.getcwd(), workers_per_node=12)
        executor = ParslPoolExecutor(config)
        launcher = ExecutorLauncher(executor, close_exchange=True)


    if args.log_dir != "agent_logs":
        logpath = args.log_dir
    else:
        run_id = len(os.listdir(args.log_dir))
        logpath = f"{args.log_dir}/{run_id:03}"

    os.makedirs(logpath, exist_ok=True)

    tracker_handle = spawn_tracker(thread_launcher, exchange)
    spawn_agents(adj_dict, exchange=exchange,
                 launcher=launcher,
                 logpath=logpath,
                 tracker_handle=tracker_handle,
                 rounds=int(args.rounds),
                 model_size=args.model_size)

    logger.info("Waiting for tracker")
    thread_launcher.wait(tracker_handle.agent_id)
    logger.info("Tracker done")


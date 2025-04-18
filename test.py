from __future__ import annotations

import aeris
import random
import numpy as np
import argparse
import logging
import time

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
from aeris.launcher.thread import ThreadLauncher
from concurrent.futures import ThreadPoolExecutor
from aeris.launcher.executor import ExecutorLauncher
from aeris.logging import init_logging
from parsl_config import PARSL_CONFIGS

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


class DMLAgent(Behavior):

    def __init__(self, node_id: int, neighbors: HandleList[DMLAgent],
                 rounds: int = 2) -> None:
        init_logging(logging.INFO, logfile=f"agent_logs/agent.{node_id}.log")
        self.node_id = node_id
        self.neighbors = neighbors
        self.rounds = rounds
        self.inbox = []
        self.state = random.randint(1, 100)
        logger.warning(f"[{self.node_id}] Init with {self.neighbors}")

    def train(self):

        logger.warning(f"[{self.node_id}] Train loop {self.state=} {self.inbox=}")
        self.state += sum(self.inbox)
        self.inbox = []
        time.sleep(1)

    def push_state(self):
        for neighbor in self.neighbors:
            neighbor.action('receive_state', self.state, self.node_id).result()

    @action
    def receive_state(self, state:int, from_id: int) -> None:
        logger.info(f"[{self.node_id}] Received {state=} {from_id=}")
        self.inbox.append(state)
        return

    @loop
    def training_loop(self, shutdown: threading.Event) -> None:
        while not shutdown.is_set() and self.rounds > 0:
            self.train()
            self.push_state()
            self.rounds -= 1
        logger.warning("Ready to exit")
        time.sleep(5)
        # Sort of crappy way to exit
        shutdown.set()



def spawn_agents(adj_dict: dict[list], exchange, launcher: ExecutorLauncher) -> None:

    nodes = {}

    # First create handles for all nodes
    for node in adj_dict:
        node_id = exchange.create_agent()
        node_handle: UnboundRemoteHandle[DMLAgent] = exchange.create_handle(
            node_id)
        #node_handle == exchange.create_handle(node_id)
        nodes[node] = {'node_id' : node_id,
                       'node_handle': node_handle}

    print(f"{nodes=}")

    # Create behaviors with handles to its neighbors
    for node in nodes:
        print(f"{node=} {nodes[node]=}")
        neighbor_handles = [nodes[n]['node_handle'] for n in adj_dict[node]]
        print(neighbor_handles)

        node_behavior = DMLAgent(node_id=node,
                                 neighbors=HandleList(neighbor_handles))

        agent_handle = launcher.launch(
            node_behavior,
            exchange,
            agent_id=nodes[node]['node_id']
        )

        nodes[node]['agent_handle'] = agent_handle

    logger.info("Finished launching agents")
    for node in nodes:
        agent_id = nodes[node]['node_id']
        print(f"Waiting for {agent_id=}")
        launcher.wait(agent_id)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--topo_file", required=True,
                        help="Topology file")
    parser.add_argument("-e", "--environment", default='threads',
                        help="environment threads/parsl")
    parser.add_argument('--redis_hostname',
                        help="redis hostname")
    parser.add_argument('--redis_port', default=6789,
                        help="redis port")
    args = parser.parse_args()


    adj_dict = get_adj_dict(args.topo_file)

    print(adj_dict)

    print(f"Running in {args.environment}")
    if args.environment == 'threads':
        exchange = ThreadExchange()
        executor = ThreadPoolExecutor()
        launcher = ExecutorLauncher(executor)
    elif args.environment == 'parsl-gpu':
        exchange = RedisExchange(hostname=args.redis_hostname, port=args.redis_port)

        config = PARSL_CONFIGS['htex-aurora-gpu'](run_dir=os.getcwd(), workers_per_node=12)
        executor = ParslPoolExecutor(config)
        launcher = ExecutorLauncher(executor, close_exchange=True),

    spawn_agents(adj_dict, exchange, launcher)

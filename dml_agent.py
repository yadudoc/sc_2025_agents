from __future__ import annotations

import aeris
import random
import numpy as np
import argparse
import logging
import time
import os
import sys
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
from decentralized import CNN, Node, calculate_model_size
from parsl_config import PARSL_CONFIGS
import pickle
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

    def __init__(self, node_id: int,
                 neighbors: HandleList[DMLAgent],
                 tracker: Handle[Tracker],
                 logpath: str,
                 rounds: int = 4,
                 model_size: str = 'medium',
                 epochs_per_round: int = 1) -> None:

        self.node_id = node_id
        self.neighbors = neighbors
        self.rounds = rounds
        self.current_round = 0
        self.inbox = []
        self.state = random.randint(1, 100)
        self.epochs_per_round = epochs_per_round
        self.logpath = logpath
        self.tracker = tracker
        self.model_size = model_size
        logger.info(f"[{self.node_id}] Initalized")
        logger.info(f"[{self.node_id}] {rounds=}")
        logger.info(f"[{self.node_id}] {model_size=}")

    def on_setup(self):
        init_logging(logging.INFO, logfile=f"{self.logpath}/agent.{self.node_id}.log", color=False, extra=True)
        logger.info(f"[{self.node_id}] Setup: Initializing model")
        self.init_model()

    def init_model(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.xpu.is_available():
            self.device = torch.device('xpu')
        else:
            self.device = torch.device('cpu')

        # Load and preprocess MNIST dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        # Initialize global model architecture (each node will get a copy)
        model = CNN(model_size=self.model_size).to(self.device)
        self.node = Node(
            node_id=self.node_id,
            model=model,
            train_data=loader,
            device=self.device
        )
        # self.model_state_size = sys.getsizeof(pickle.dumps(model.state_dict()))
        self.model_state_size = calculate_model_size(model)
        logger.info(f"[{self.node_id}] Initalized model")

    def train(self) -> float:
        # Local training phase
        s = time.perf_counter()
        self.node.train_local(epochs=self.epochs_per_round)
        t = time.perf_counter() - s
        logger.info(f"[{self.node_id}] Completed training for epochs:{self.epochs_per_round} in {t}s")
        return t

    def aggregate(self) -> float:
        # FedAverage
        s = time.perf_counter()
        self.node.average_with_neighbors(self.inbox)
        self.inbox = []
        t = time.perf_counter() - s
        return t

    def push_state(self) -> float:
        # TODO: We shouldn't need to ship the model entirely, but rather
        # just the state dict because the model contains back refs
        push_futures = []
        s = time.perf_counter()
        model_state = self.node.model.state_dict()
        for neighbor in self.neighbors:
            try:
                future = neighbor.action('receive_state', model_state, self.node_id)
            except aeris.exception.MailboxClosedError:
                logger.info(f"Could not ship because mailbox was closed on {neighbor}")
            push_futures.append(future)

        for fu in push_futures:
            try:
                fu.result()
            except aeris.exception.MailboxClosedError:
                logger.info("Pass MailboxClosed")
            except Exception:
                logger.exception("Something broke")
        t = time.perf_counter() - s
        logger.info(f"[{self.node_id}] pushed state size:{self.model_state_size  / 8e6:.2f} MB to neighbors:{len(self.neighbors)} in {t}s")
        return t

    @action
    def receive_state(self, state:int, from_id: int) -> None:
        self.inbox.append(state)
        return

    @loop
    def training_loop(self, shutdown: threading.Event) -> None:
        logger.info("Starting loop")
        s = time.perf_counter()

        t_tot_train, t_tot_push, t_tot_aggregate = 0, 0, 0
        while not shutdown.is_set() and self.current_round < self.rounds:
            logger.info(f"[{self.node_id}] Round:{self.current_round}")
            train_t = self.train()
            push_t = self.push_state()
            agg_t = self.aggregate()

            t_tot_train += train_t
            t_tot_push += push_t
            t_tot_aggregate += agg_t

            self.current_round += 1

        t = time.perf_counter() - s
        logger.info(f"[{self.node_id}] total_train_t:{t_tot_train} total_push_t:{t_tot_push} total_agg_t:{t_tot_aggregate}")

        # Report to tracker that the agent is done
        self.tracker.action('report_done', self.node_id).result()
        logger.info(f"[{self.node_id}] Exiting. Total active time: {t}s")


class Tracker(Behavior):

    def __init__(self, agent_count: int):
        self.agent_count = agent_count
        self.done_count = 0
        self.done_event = threading.Event()

    @action
    def report_done(self, agent_id: int) -> int:
        logger.info(f"[Tracker] {agent_id} reports done")
        self.done_count += 1
        return self.done_count
        

    @action
    def block_until_done(self):
        return self.done_event.wait()

    @loop
    def wait_for_all(self, shutdown: threading.Event) -> None:
        while self.done_count != self.agent_count:
            time.sleep(1)
        self.done_event.set()
        time.sleep(1)
        logger.info("[Tracker] Exiting !!!")
        shutdown.set()



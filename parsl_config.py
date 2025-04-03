from __future__ import annotations

import os

from parsl.addresses import address_by_hostname
from parsl.addresses import address_by_interface
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.launchers import MpiExecLauncher
from parsl.providers import LocalProvider


def get_htex_local_config(
    run_dir: str,
    workers_per_node: int,
) -> Config:
    executor = HighThroughputExecutor(
        label='htex-local',
        max_workers_per_node=workers_per_node,
        address=address_by_hostname(),
        cores_per_worker=1,
        provider=LocalProvider(init_blocks=1, max_blocks=1),
    )
    return Config(
        executors=[executor],
        run_dir=run_dir,
        initialize_logging=False,
    )


def get_htex_aurora_cpu_config(
    run_dir: str,
    workers_per_node: int,
) -> Config:
    # Get the number of nodes:
    node_file = os.getenv('PBS_NODEFILE')
    with open(node_file, 'r') as f:
        node_list = f.readlines()
        num_nodes = len(node_list)

    executor = HighThroughputExecutor(
        max_workers_per_node=workers_per_node,
        # Increase if you have many more tasks than workers
        prefetch_capacity=0,
        # Options that specify properties of PBS Jobs
        provider=LocalProvider(
            # Number of nodes job
            worker_init=f'''source $HOME/ ; export PYTHONPATH=/home/yadunand/sc_2025_agents:$PYTHONPATH''',
            nodes_per_block=num_nodes,
            launcher=MpiExecLauncher(
                bind_cmd='--cpu-bind',
                overrides='--ppn 1',
            ),
            init_blocks=1,
            max_blocks=1,
        ),
    )

    return Config(
        executors=[executor],
        run_dir=run_dir,
        initialize_logging=False,
    )

def get_htex_aurora_local_config(
    run_dir: str,
    workers_per_node: int,
) -> Config:
    executor = HighThroughputExecutor(
        label='htex-local',
        max_workers_per_node=workers_per_node,
        address=address_by_interface('hsn0'),
        cores_per_worker=1,
        provider=LocalProvider(init_blocks=1, max_blocks=1),
    )
    return Config(
        executors=[executor],
        run_dir=run_dir,
        initialize_logging=False,
    )


def get_htex_aurora_gpu_config(
    run_dir: str,
    workers_per_node: int,
) -> Config:
    # Get the number of nodes:
    node_file = os.getenv('PBS_NODEFILE')
    with open(node_file, 'r') as f:
        node_list = f.readlines()
        num_nodes = len(node_list)

    tile_names = [f'{gid}.{tid}' for gid in range(6) for tid in range(2)]

    executor = HighThroughputExecutor(
        max_workers_per_node=workers_per_node,
        # Increase if you have many more tasks than workers
        prefetch_capacity=0,
        # Distributes threads to workers/tiles in a way optimized for Aurora
        # cpu_affinity="list:1-8,105-112:9-16,113-120:17-24,121-128:25-32,129-136:33-40,137-144:41-48,145-152:53-60,157-164:61-68,165-172:69-76,173-180:77-84,181-188:85-92,189-196:93-100,197-204",
        # GPU spec

        # tile names do not work with pytorch.
        #  available_accelerators=tile_names,
        available_accelerators=12,
        # Options that specify properties of PBS Jobs
        provider=LocalProvider(
            # Number of nodes job
            nodes_per_block=num_nodes,
            worker_init=f'''source $HOME/ ; export PYTHONPATH=/home/yadunand/sc_2025_agents:$PYTHONPATH''',
            launcher=MpiExecLauncher(
                bind_cmd='--cpu-bind',
                overrides='--ppn 1',
            ),
            init_blocks=1,
            max_blocks=1,
            min_blocks=1,
        ),
    )

    return Config(
        executors=[executor],
        run_dir=run_dir,
        initialize_logging=False,
    )


PARSL_CONFIGS = {
    'htex-local': get_htex_local_config,
    'htex-aurora-cpu': get_htex_aurora_cpu_config,
    'htex-aurora-local': get_htex_aurora_local_config,
    'htex-aurora-gpu': get_htex_aurora_gpu_config,
}

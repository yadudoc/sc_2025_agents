# config.py
import os
from parsl.config import Config
import parsl
from parsl import python_app

# Use LocalProvider to launch workers within a submitted batch job
from parsl.providers import LocalProvider
# The high throughput executor is for scaling large single core/tile/gpu tasks on HPC system:
from parsl.executors import HighThroughputExecutor
# Use the MPI launcher to launch worker processes:
from parsl.launchers import MpiExecLauncher

# tile_names = [f'{gid}.{tid}' for gid in range(6) for tid in range(2)]

# The config will launch workers from this directory
execute_dir = os.getcwd()

# Get the number of nodes:
node_file = os.getenv("PBS_NODEFILE")
with open(node_file,"r") as f:
    node_list = f.readlines()
    num_nodes = len(node_list)


aurora_single_tile_config = Config(
    executors=[
    HighThroughputExecutor(
        # Ensures one worker per GPU tile on each node
        available_accelerators=12,
        max_workers_per_node=12,
        # Distributes threads to workers/tiles in a way optimized for Aurora
        cpu_affinity="list:1-8,105-112:9-16,113-120:17-24,121-128:25-32,129-136:33-40,137-144:41-48,145-152:53-60,157-164:61-68,165-172:69-76,173-180:77-84,181-188:85-92,189-196:93-100,197-204",
        # Increase if you have many more tasks than workers
        prefetch_capacity=0,
        # Options that specify properties of PBS Jobs
        provider=LocalProvider(
            worker_init="source ~/setup_agents.sh",
            # Number of nodes job
            nodes_per_block=num_nodes,
            launcher=MpiExecLauncher(bind_cmd="--cpu-bind", overrides="--ppn 1"),
            init_blocks=1,
            max_blocks=1,
        ),
    ),
    ],
)

@python_app
def bar():
    import torch
    return torch.xpu.is_available(), torch.xpu.device_count(), torch.xpu.current_device()

@python_app
def foo():
    import os
    import time
    time.sleep(4)
    import torch
    torch.xpu.set_device(os.environ['ZE_AFFINITY_MASK'])
    return torch.xpu.is_available(), torch.xpu.device_count(), torch.xpu.current_device(), os.environ['ZE_AFFINITY_MASK']


if __name__ == "__main__":

    parsl.load(aurora_single_tile_config)

    futures = [foo() for i in range(24)]

    for fu in futures:
        print(fu.result())

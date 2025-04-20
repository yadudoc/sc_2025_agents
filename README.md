# sc_2025_agents

Experiments for the Distributed Machine Learning application for SC 2025 agents paper

# Running the experiment

# Setup the environment for the experiments on aurora with

```bash

    # Load system environment
    module load frameworks

    # Create virtualenv
    python3 -m venv ~/agents_venv --system-site-packages

    # Install aeris from https://github.com/proxystore/aeris
    git clone git@github.com:proxystore/aeris.git
    cd aeris
    pip install .

    # Install proxy store with
    pip install proxystore proxystore-ex
```


# Launch the experiments

sbatch round3_batch_runners/runner.128.node.sh

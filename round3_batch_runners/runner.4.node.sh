#!/bin/bash
#PBS -A AuroraGPT
#PBS -N agents-action-decentral
#PBS -q debug-scaling
#PBS -l select=4
#PBS -l walltime=01:00:00
#PBS -l filesystems=flare
#PBS -l place=scatter
#PBS -j oe
#PBS -o /home/yadunand/sc_2025_agents/round3_batch_runners/node.4.stdout
#PBS -e /home/yadunand/sc_2025_agents/round3_batch_runners/node.4.stderr
#PBS -M yadudoc1729@gmail.com
#PBS -m e

set NUMEXPR_MAX_THREADS=272

echo "Loading env from ~/setup_agents.sh"
cd /home/yadunand/sc_2025_agents
source ~/setup_agents.sh

NODES=$(wc -l < $PBS_NODEFILE)
###############
# Start REDIS #
###############

REDIS_PORT=6389
REDIS_HOST=$(ip -f inet addr show hsn0 | awk '/inet / {print $2}' | cut -d/ -f1)
/home/yadunand/.conda/envs/redis/bin/redis-server --port $REDIS_PORT --save "" --appendonly no --protected-mode no &> /dev/null &
REDIS=$!
echo "Redis server started on $REDIS_HOST @ $REDIS_PORT"


###################
# Run AERIS tests #
###################

AGENTS=$(($NODES * 12))

echo "Running tests for AGENTS=$AGENTS on NODES=$NODES"
LOGDIR=/home/yadunand/sc_2025_agents/experiment_3

for GRAPHTYPE in "powerlaw"
do
    for model_size in "small" "medium" "large" "largex2"
    do
        EXP_LOG=$LOGDIR/$GRAPHTYPE.$AGENTS.$model_size

        if [ ! -d $EXP_LOG ]
        then
            echo "$EXP_LOG is not computed. Running"
        else
            echo "$EXP_LOG is present. Skipping experiment"
	        continue
        fi

        if [ "$model_size" = "small" ]
        then
            echo "Logging experiment to $EXP_LOG.round1"
            timeout -k 10 1200 python3 fedml_test.py \
                    -t topology/topo_${GRAPHTYPE}_${AGENTS}.txt \
                    -e parsl-gpu-proxy \
                    --redis_hostname=$REDIS_HOST \
                    --redis_port=$REDIS_PORT \
                    --log_dir=$EXP_LOG.round1 \
                    --model_size=$model_size
        fi

        echo "Logging experiment to $EXP_LOG"
        timeout -k 10 1200 python3 fedml_test.py \
                -t topology/topo_${GRAPHTYPE}_${AGENTS}.txt \
                -e parsl-gpu-proxy \
                --redis_hostname=$REDIS_HOST \
                --redis_port=$REDIS_PORT \
                --log_dir=$EXP_LOG \
                --model_size=$model_size
    done
done

kill $REDIS
echo "Redis server stopped"

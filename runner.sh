#!/bin/bash -e
#PBS -A AuroraGPT
#PBS -N agents-action-decentral
#PBS -q debug
#PBS -l select=2
#PBS -l walltime=00:60:00
#PBS -l filesystems=flare
#PBS -l place=scatter
#PBS -j oe


#############
# RUN AERIS #
#############

REDIS_PORT=6389
REDIS_HOST=$(hostname -f)
/home/yadunand/.conda/envs/redis/bin/redis-server --port $REDIS_PORT --save "" --appendonly no --protected-mode no &> /dev/null &
REDIS=$!
echo "Redis server started"

which python3
python3 fedml_test.py \
        -t topology/topo_2.txt \
        -e parsl-gpu \
        --redis_hostname=$REDIS_HOST \
        --redis_port=$REDIS_PORT

kill $REDIS
echo "Redis server stopped"

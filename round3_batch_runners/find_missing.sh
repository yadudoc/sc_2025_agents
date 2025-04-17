#!/bin/bash

EXP="/home/yadunand/sc_2025_agents/experiment_3"

for AGENTCOUNT in 12 24 48 96 192 384 768 1536
do
    for TOPOLOGY in "powerlaw"
    do
        for SIZE in "small" "medium" "large" "largex2"
        do
            EXP_DIR=$EXP/$TOPOLOGY.$AGENTCOUNT.$SIZE
            if [  -d $EXP_DIR ]
            then
                DATA_POINT_COUNT=$(grep -R "total_train" $EXP_DIR | wc -l)
                if [ "$DATA_POINT_COUNT" == "$AGENTCOUNT" ]
                then
                    # echo "$EXP_DIR IS COMPLETE"
                    true
                else
                    echo "$EXP_DIR IS NOT COMPLETE"
                    # echo "Wiping $EXP_DIR"
                    # rm -rf $EXP_DIR
                fi
            else
                echo "$EXP_DIR IS MISSING"
            fi
        done
    done
done

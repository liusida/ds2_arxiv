#!/bin/sh
set -x
if true
then
    for seed in 60 61 62 63 64 65 66 67 68 69
    do
        sbatch -J e7  ~/bin/bluemoon.sh ds2 python ds2_arxiv/9.2.new_algorithm_scheduling.py --tag=e7 --seed=$seed --epoch_steps=1e4 --num_epochs=1e3
    done
fi
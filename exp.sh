#!/bin/sh
set -x
if true
for seed in 0 1 2 3 4 5 6 7 8 9
do
    sbatch -J e7  ~/bin/bluemoon.sh ds2 python ds2_arxiv/9.2.new_algorithm_scheduling.py --tag=e7 --seed=$seed --epoch_steps=1e4 --num_epochs=1e3
done

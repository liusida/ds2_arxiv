#!/bin/sh
set -x
if true
then
    expName="ParallelHillClimber"
    for seed in 70 71 72 73 74 75 76 77 78 79
    do
        sbatch -J $expName ~/bin/deepgreen.sh ds2 python ds2_arxiv/10.0.ea.parallel_hillclimber.py --tag=$expName --seed=$seed --pop_size=1000 -n=100
    done
fi

if false
then
    for seed in 70 71 72 73 74 75 76 77 78 79
    do
        sbatch -J gpu93  ~/bin/deepgreen.sh ds2 python ds2_arxiv/9.3.new_algo_on_gpu.py --tag=gpu93 --seed=$seed --num_epochs=1e4
    done
fi

if false
then
    for seed in 60 61 62 63 64 65 66 67 68 69
    do
        sbatch -J e7  ~/bin/bluemoon.sh ds2 python ds2_arxiv/9.2.new_algorithm_scheduling.py --tag=e7 --seed=$seed --epoch_steps=1e4 --num_epochs=1e3
    done
fi
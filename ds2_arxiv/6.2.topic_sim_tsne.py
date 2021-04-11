import csv, re, pickle, argparse
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd

topic_sim = np.load("shared/topic_similarity_matrix.npy")
total_length = topic_sim.shape[0]

# Sanity check
for i in range(total_length):
    assert topic_sim[i,i] > 0.999, "Should be the same with it self."
    j = np.random.randint(low=0, high=total_length)
    assert topic_sim[i,j] == topic_sim[j,i], "Should be symmetric."

# Record the similarity matrix before t-SNE
plt.imshow(topic_sim)
plt.colorbar()
plt.savefig("tmp_before_sort.png")
plt.close()


def tsne(perplexity, learning_rate, run, verbose=False, visual_check=False):
    # return 0,0,1,0.1 # Mock for debug
    print(f"start tsne with ({perplexity}, {learning_rate}, {run})")
    ret = TSNE(n_components=1, perplexity=perplexity, learning_rate=learning_rate, n_iter=3000, verbose=verbose).fit(topic_sim)
    
    # https://arxiv.org/pdf/1708.03229.pdf
    s_score = 2 * ret.kl_divergence_ + np.log(topic_sim.shape[0]) * perplexity / topic_sim.shape[0]

    print(f"perplexity={perplexity}, learning_rate={learning_rate}, run={run}, kl_divergence_={ret.kl_divergence_}, s_score={s_score}")
    indices = np.argsort(ret.embedding_.flatten())

    cos_sim_after = topic_sim
    cos_sim_after = cos_sim_after[indices].T
    cos_sim_after = cos_sim_after[indices]

    if visual_check: # visual check
        plt.imshow(cos_sim_after)
        plt.colorbar()
        plt.savefig(f"tmp_after_sort_{perplexity}_{learning_rate}_{run}.png")
        plt.close()
    return indices, cos_sim_after, ret.kl_divergence_, s_score

if __name__ == '__main__':
    # after sweep for hyperparameters, perplexity=500, learning_rate=10 is the best.
    if True:
        jobs = []
        for perplexity in [64, 128, 256, 512, 1024, 2048]:
            for learning_rate in [10, 20, 40, 80, 160]:
                for run in range(3):
                    jobs.append([perplexity,learning_rate,run])
        sweep_results = []
        df = pd.DataFrame(columns=['perplexity', 'learning_rate', 'run', 'kldivergence', 's_score'])
        for job in jobs:
            perplexity,learning_rate,run = job
            _, _, kldivergence, s_score = tsne(*job)
            df = df.append({
                'perplexity': perplexity, 'learning_rate': learning_rate, 'run': run, 'kldivergence': kldivergence, 's_score': s_score,
            }, ignore_index=True)
        df.to_pickle("shared/sweep_sscore_results.df")
        print(df)

    # indices,_,_,_ = tsne(512,20,0,1,1)
    # np.save("shared/topic_similarity_indices.npy", indices)

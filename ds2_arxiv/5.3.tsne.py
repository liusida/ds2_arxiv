import csv, re, pickle, argparse
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=66, help="DG can support 240") # 4422 = 66*67
args = parser.parse_args()

# Read Features
features = torch.load("data/features/features_bert_topics.pt") # 4422 x 768
features = features.numpy()
print(f"features.shape {features.shape}")

# Read detailed information
with open("shared/cited_100.pickle", "rb") as f:
    data = pickle.load(f)

# Sort from highest cited to lowest
indices = np.argsort(data['cites'])[::-1]

categories = np.array(data['categories'])[indices]
titles = np.array(data['titles'])[indices]
titles = [re.sub(r'\s+', ' ', t) for t in titles]
arxiv_ids = np.array(data['arxiv_ids'])[indices]
features = features[indices]

# Compute similarity for t-SNE
features = torch.Tensor(features)
batch_size = args.batch_size # 60 is due to the limitation of the GPU memory
total_length = features.shape[0]
total_batch = int(total_length/batch_size) # throw away the last batch if last batch is not a full batch
full_cos_sim = np.zeros(shape=[total_length, total_length], dtype=np.float32)
for i in range(total_batch):
    batch_features_i = features[i*batch_size: (i+1)*batch_size]
    for j in range(total_batch):
        batch_features_j = features[j*batch_size: (j+1)*batch_size]
        cos_sim = torch.nn.functional.cosine_similarity(batch_features_i[:,:,None], batch_features_j.t()[None,:,:])
        cos_sim = np.array(cos_sim).astype(np.float32)
        full_cos_sim[i*batch_size: (i+1)*batch_size, j*batch_size: (j+1)*batch_size] = cos_sim
        if False: # visual check
            plt.imshow(full_cos_sim)
            plt.savefig(f"tmp_{i}_{j}.png")
            plt.close()
print(full_cos_sim.shape)

# Sanity check
for i in range(total_length):
    assert full_cos_sim[i,i] > 0.999, "Should be the same with it self."
    j = np.random.randint(low=0, high=total_length)
    assert full_cos_sim[i,j] == full_cos_sim[j,i], "Should be symmetric."

# Record the similarity matrix before t-SNE
plt.imshow(full_cos_sim)
plt.colorbar()
plt.savefig("tmp_before_sort.png")
plt.close()


def tsne(perplexity, learning_rate, run):
    # return 0,0,1,0.1 # Mock for debug
    print(f"start tsne with ({perplexity}, {learning_rate}, {run})")
    ret = TSNE(n_components=1, perplexity=perplexity, learning_rate=learning_rate, n_iter=3000, verbose=0).fit(full_cos_sim)
    
    # https://arxiv.org/pdf/1708.03229.pdf
    s_score = 2 * ret.kl_divergence_ + np.log(full_cos_sim.shape[0]) * perplexity / full_cos_sim.shape[0]

    print(f"perplexity={perplexity}, learning_rate={learning_rate}, run={run}, kl_divergence_={ret.kl_divergence_}, s_score={s_score}")
    indices = np.argsort(ret.embedding_.flatten())

    cos_sim_after = full_cos_sim
    cos_sim_after = cos_sim_after[indices].T
    cos_sim_after = cos_sim_after[indices]

    if False: # visual check
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
            for learning_rate in [10, 20, 40]:
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


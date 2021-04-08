import glob, pickle, argparse
from smart_open import open
import xmltodict
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import FeatureExtractionPipeline
import wandb
wandb.init("DS2")

parser = argparse.ArgumentParser()
parser.add_argument("--skip", type=int, default=0)
args = parser.parse_args()

if args.skip<=0:
    arxiv_ids = []
    abstracts = []
    categories = []
    filenames = glob.glob("data/harvest_LG_AI/*.xml")
    l = len(filenames)
    for i, filename in enumerate(filenames):
        with open(filename, "r") as f:
            record_xml = f.read()
        if len(record_xml)<10:
            # must be bad record
            print(f"Bad record {filename}")
            continue
        record_dict = xmltodict.parse(record_xml, process_namespaces=False)['record']['metadata']['arXiv']
        arxiv_id = filename.split("/")[-1].split(".xml")[0]
        category = record_dict['categories']
        abstract = record_dict['abstract']

        abstract = abstract.replace("\n", " ")
        arxiv_ids.append(arxiv_id)
        categories.append(category)
        abstracts.append(abstract)
        wandb.log({f"read_step_{l}": i})
    with open("data/features/input.pickle", "wb") as f:
        pickle.dump([arxiv_ids, categories,abstracts], f) # size: O(N)

if args.skip<=1:
    with open("data/features/input.pickle", "rb") as f:
        arxiv_ids, categories,abstracts = pickle.load(f)

    corpus = abstracts 

    batch_size = 60 # 60 is due to the limitation of the GPU memory
    total_length = len(corpus)
    total_batch = int((total_length-1)/batch_size) # throw away the last little bit

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_id = 0 if torch.cuda.is_available() else -1 # only support one GPU
    print('Using device:', device)
    print()

    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pp = FeatureExtractionPipeline(model=model, tokenizer=tokenizer, device=device_id) # 0: use GPU:0

    with torch.no_grad():
        rets = []
        for i in range(total_batch):
            batch_corpus = corpus[i*batch_size: (i+1)*batch_size]
            inputs = pp.tokenizer(batch_corpus, return_tensors=pp.framework, padding='max_length', max_length=512, truncation=True)
            inputs = pp.ensure_tensor_on_device(**inputs)
            ret = pp.model.bert(**inputs) # I only stepped into this model: "nlptown/bert-base-multilingual-uncased-sentiment"
            # There are two choices: output_1 is the whole activation chain, output_2 is the processed first cell.
            # I feel that output_1 contains much richer information, but the dimension varies according to the length of the document.
            output_1 = torch.flatten(ret.last_hidden_state, start_dim=1)
            output_2 = ret.pooler_output
            output_3 = torch.cat([ret.last_hidden_state[:,0],ret.last_hidden_state[:,-1]], dim=1)
            rets.append(output_1.cpu())
            if i==0:
                print(f"ret.last_hidden_state.shape: {ret.last_hidden_state.shape}\nret.pooler_output.shape: {ret.pooler_output.shape}")
            wandb.log({f"BERT_batch_{total_batch}": i})
        torch.save(rets, "data/features/BERT.pt") # size: O(N) x 512 x 768
if args.skip<=2:
    with torch.no_grad():
        rets = torch.load("data/features/BERT.pt")

        cos_sim = np.zeros([total_batch*batch_size, total_batch*batch_size])
        total_loop = int(len(rets) * (len(rets)+1) / 2)
        for i in range(len(rets)):
            for j in range(i+1):
                ret = torch.nn.functional.cosine_similarity(rets[i][:,:,None].to(device), rets[j].t()[None,:,:].to(device))
                if i==j:
                    ret = torch.tril(ret)
                cos_sim[i*batch_size:(i+1)*batch_size, j*batch_size:(j+1)*batch_size] = ret.cpu().numpy()
                wandb.log({f"cosine_similarity_step_{total_loop}": i})
        with open("data/features/BERT_pairwise_compare.np", "wb") as f:
            np.save(f, cos_sim) # size: O(N x N)
if args.skip<=3:
    with open("data/features/BERT_pairwise_compare.np", "rb") as f:
        cos_sim = np.load(f)

    np.fill_diagonal(cos_sim, 0)
    max_similar = np.argmax(cos_sim)
    max_similar_i = max_similar // len(cos_sim)
    max_similar_j = max_similar % len(cos_sim)

    print(cos_sim)
    plt.imshow(cos_sim)
    plt.colorbar()
    plt.savefig("tmp_3.png")

    with open("data/features/input.pickle", "rb") as f:
        arxiv_ids, categories,abstracts = pickle.load(f)

    print(f"most similar pair of papers {max_similar_i} and {max_similar_j} (score={cos_sim[max_similar_i, max_similar_j]})")
    print(arxiv_ids[max_similar_i])
    print(categories[max_similar_i])
    print(abstracts[max_similar_i])
    print()
    print(arxiv_ids[max_similar_j])
    print(categories[max_similar_j])
    print(abstracts[max_similar_j])

import re, pickle, argparse
from smart_open import open
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import FeatureExtractionPipeline

parser = argparse.ArgumentParser()
parser.add_argument("--skip", type=int, default=0)
parser.add_argument("--batch-size", type=int, default=66, help="DG can support 240") # 4422 = 66*67
parser.add_argument("--wandb", action="store_true")
args = parser.parse_args()
if args.wandb:
    import wandb
    wandb.init("DS2")
    wandb.config.update(args)
def wandb_log(x):
    if args.wandb:
        try:
            wandb.log(x)
        except Exception as e:
            print("WandB error: ", e)
    else:
        print(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_id = 0 if torch.cuda.is_available() else -1 # only support one GPU
print('Using device:', device)
print()


if args.skip<=1:
    with open("shared/cited_100.pickle", "rb") as f:
        data = pickle.load(f)
    
    # construct descriptions for BERT to read
    #
    descriptions =[]
    arxiv_ids = []
    for i, arxiv_id in enumerate(data["arxiv_ids"]):
        abstract = data['abstracts'][i]
        abstract = re.sub(r'\s+', ' ', abstract)                 # remove unnecessary returns
        abstract = re.sub(r'\$([a-zA-Z0-9])\$', r'\1', abstract) # $d$ -> d
        abstract = re.sub(r'\$[^\$]+\$', '*', abstract)          # $xyz$ -> *
        # this time, let's exclude the abstract.
        # desc = f"This paper is about {', '.join(data['topic_lists'][i])}. {abstract}"
        desc = f"This paper is about {', '.join(data['topic_lists'][i])}."
        arxiv_ids.append(arxiv_id)
        descriptions.append(desc)

    corpus = descriptions 

    batch_size = args.batch_size # 60 is due to the limitation of the GPU memory
    total_length = len(corpus)
    total_batch = int(total_length/batch_size) # throw away the last batch if last batch is not a full batch
    with open("data/features/settings.pickle", "wb") as f:
        pickle.dump([batch_size, total_length, total_batch], f)
    print("settings saved.", flush=True)

    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pp = FeatureExtractionPipeline(model=model, tokenizer=tokenizer, device=device_id) # 0: use GPU:0

    with torch.no_grad():
        # measure the max length of the input sequences
        max_length = 512
        inputs = pp.tokenizer(corpus, return_tensors=pp.framework, padding='max_length', max_length=512, truncation=True)
        for i in range(512):
            match = (inputs['input_ids'][:,i] != 0).sum()
            if match==0:
                print(f"max length {i}")
                max_length = i
                break

        rets = []
        for i in range(total_batch):
            batch_corpus = corpus[i*batch_size: (i+1)*batch_size]
            inputs = pp.tokenizer(batch_corpus, return_tensors=pp.framework, padding='max_length', max_length=max_length, truncation=True)
            if True:
                print(f"Original: \n{batch_corpus[0]}")
                print(f"Transformed:")
                print(' '.join(pp.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])))
            inputs = pp.ensure_tensor_on_device(**inputs)
            ret = pp.model.distilbert(**inputs) # this is for the model: "distilbert-base-uncased-finetuned-sst-2-english"
            ret = ret[0][:, 0].cpu() # get the raw first activation. I don't know why BERT model will do dropout during test.

            # previously I used "nlptown/bert-base-multilingual-uncased-sentiment"
            # ret = pp.model.bert(**inputs) # I only stepped into this model: "nlptown/bert-base-multilingual-uncased-sentiment"
            # # There are two choices: output_1 is the whole activation chain, output_2 is the processed first cell.
            # # I feel that output_1 contains much richer information, but the dimension varies according to the length of the document.
            # output_1 = torch.flatten(ret.last_hidden_state, start_dim=1)
            # output_2 = ret.pooler_output
            # output_3 = torch.cat([ret.last_hidden_state[:,0],ret.last_hidden_state[:,-1]], dim=1)
            # ret = output_2.cpu()

            rets.append(ret)
            if i==0:
                print(f"rets[-1].shape = {rets[-1].shape}", flush=True)
            print(f"BERT_batch_{total_batch} : {i+1}", flush=True)
            wandb_log({f"BERT_batch_{total_batch}": i+1})

        rets = torch.cat(rets, dim=0)
        torch.save(rets, "data/features/features_bert_topics.pt") # size: O(N) x 768

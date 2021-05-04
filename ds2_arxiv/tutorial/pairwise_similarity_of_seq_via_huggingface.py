# Author: Sida Liu (learner.sida.liu@gmail.com), 2020
# Reference: 
#   https://huggingface.co/transformers/master/quicktour.html
#   https://huggingface.co/transformers/master/main_classes/feature_extractor.html
import torch
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import FeatureExtractionPipeline

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
pp = FeatureExtractionPipeline(model=model, tokenizer=tokenizer, device=0) # 0: use GPU:0

# Each line in the corpus is a document, here are 4 examples:
corpus = """
Hello, the world!
Nice to meet you!
Dimension where cosine similarity is computed.
Small value to avoid division by zero.
""".strip().split("\n")

with torch.no_grad():
    inputs = pp._parse_and_tokenize(corpus)
    inputs = pp.ensure_tensor_on_device(**inputs)
    ret = pp.model.bert(**inputs) # I only stepped into this model: "nlptown/bert-base-multilingual-uncased-sentiment"
    # There are two choices: output_1 is the whole activation chain, output_2 is the processed first cell.
    # I feel that output_1 contains much richer information, but the dimension varies according to the length of the document.
    output_1 = torch.flatten(ret.last_hidden_state, start_dim=1)
    output_2 = ret.pooler_output
    x = output_1
    print(f"ret.last_hidden_state.shape: {ret.last_hidden_state.shape}\nret.pooler_output.shape: {ret.pooler_output.shape}\nx.shape: {x.shape}\n")

    ret = torch.nn.functional.cosine_similarity(x[:,:,None], x.t()[None,:,:])
    ret = ret.detach().cpu().numpy()
print(ret)
plt.imshow(ret)
plt.colorbar()
plt.savefig("tmp_1.png")
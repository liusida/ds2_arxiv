import torch

features = torch.load("data/features/BERT_features.pt") # 4422 x 768
features = features.numpy()
print(features.shape)
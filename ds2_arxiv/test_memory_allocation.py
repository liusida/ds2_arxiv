import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x = torch.rand(size=[128, 393216])
print(x.shape)
ret = torch.nn.functional.cosine_similarity(x[:,:,None].to(device), x.t()[None,:,:].to(device))
print(ret.shape)
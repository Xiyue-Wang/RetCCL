import torchvision
import torch
import os
model = torchvision.models.__dict__['resnet50'](pretrained=False)
state_dict = torch.load(r'./best_ckpt.pth')

for k in list(state_dict.keys()):
    # retain only encoder_q up to before the embedding layer
    if k.startswith('encoder_q') and not k.startswith('encoder_q.fc'):
        # remove prefix
        state_dict[k[len("encoder_q."):]] = state_dict[k]
    # delete renamed or unused k
    del state_dict[k]

msg = model.load_state_dict(state_dict, strict=False)
print("Missed keys: ", msg.missing_keys)
assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
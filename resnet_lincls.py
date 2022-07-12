import torchvision
import torch
import os
model = torchvision.models.__dict__['resnet50'](pretrained=False)
state_dict = torch.load(r'./best_ckpt.pth')

for name, param in model.named_parameters():
    if name not in ['fc.weight', 'fc.bias']:
        param.requires_grad = False
# init the fc layer
model.fc.weight.data.normal_(mean=0.0, std=0.01)
model.fc.bias.data.zero_()


msg = model.load_state_dict(state_dict, strict=False)
print("Missed keys: ", msg.missing_keys)
assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
assert len(parameters) == 2  # fc.weight, fc.bias
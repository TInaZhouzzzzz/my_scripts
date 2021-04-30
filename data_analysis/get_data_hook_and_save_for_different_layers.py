import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

net = models.resnet18()

#2: 特征提取
class FeatureExtractor(nn.Module):
    def __init__(self, model, layers):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_inputs_hook(layer_id))

    def save_inputs_hook(self, layer_id):
        def fn(_, input, output):
            self._features[layer_id] = input
        return fn

    def forward(self, x):
        _ = self.model(x)
        return self._features
 
modules= net.named_modules()
layers=[]
for name, module in modules:
    if isinstance(module, nn.Conv2d) or  isinstance(module, nn.Linear):
        layers.append(name)

x = torch.rand((2, 3, 224, 224))
resnet_features = FeatureExtractor(net, layers)
features = resnet_features(x)

print(features)
print(len(features))
for i in features:
    print(i, len(features[i]))
#y=net(x)

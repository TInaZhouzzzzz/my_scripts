#1: 模型执行详情
class VerboseExecution(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

        # Register a hook for each layer
        for name, layer in self.model.named_children():
            layer.__name__ = name
            layer.register_forward_hook(
                lambda layer, _, output: print(f"{layer.__name__}: {output.shape}")
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
      
import torch
from torchvision.models import resnet50

verbose_resnet = VerboseExecution(resnet50())
dummy_input = torch.ones(10, 3, 224, 224)

_ = verbose_resnet(dummy_input)
# conv1: torch.Size([10, 64, 112, 112])
# bn1: torch.Size([10, 64, 112, 112])
# relu: torch.Size([10, 64, 112, 112])
# maxpool: torch.Size([10, 64, 56, 56])
# layer1: torch.Size([10, 256, 56, 56])
# layer2: torch.Size([10, 512, 28, 28])
# layer3: torch.Size([10, 1024, 14, 14])
# layer4: torch.Size([10, 2048, 7, 7])
# avgpool: torch.Size([10, 2048, 1, 1])
# fc: torch.Size([10, 1000])      


#2: 特征提取
class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        _ = self.model(x)
        return self._features
     
resnet_features = FeatureExtractor(resnet50(), layers=["layer4", "avgpool"])
features = resnet_features(dummy_input)

print({name: output.shape for name, output in features.items()})
# {'layer4': torch.Size([10, 2048, 7, 7]), 'avgpool': torch.Size([10, 2048, 1, 1])}    

#3: 梯度裁剪
def gradient_clipper(model: nn.Module, val: float) -> nn.Module:
    for parameter in model.parameters():
        parameter.register_hook(lambda grad: grad.clamp_(-val, val))
    
    return model

clipped_resnet = gradient_clipper(resnet50(), 0.01)
pred = clipped_resnet(dummy_input)
loss = pred.log().mean()
loss.backward()

print(clipped_resnet.fc.bias.grad[:25])
# tensor([-0.0010, -0.0047, -0.0010, -0.0009, -0.0015,  0.0027,  0.0017, -0.0023,
#          0.0051, -0.0007, -0.0057, -0.0010, -0.0039, -0.0100, -0.0018,  0.0062,
#          0.0034, -0.0010,  0.0052,  0.0021,  0.0010,  0.0017, -0.0100,  0.0021,
#          0.0020])  

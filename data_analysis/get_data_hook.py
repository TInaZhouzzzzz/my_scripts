import torch
import torch.nn as nn
import torchvision.models as models

net = models.resnet18()
print(net)

datas=[]

i = 0
def for_hook(module, input, output):
#   datas.append(input.clone().detach())
    print("here in hook")
    datas.append(input)

modules= net.named_modules()
print("======The layers in is: ========")
for name, module in modules:
    if isinstance(module, nn.Conv2d) or  isinstance(module, nn.Linear):
        print("name: ", name)
        module.register_forward_hook(for_hook)
print("here3")


x = torch.rand((2, 3, 224, 224))
y = net(x)

print("datas: ")
print(datas)
print(len(datas))

handle.remove()

import torch

def _quanz_fix1(data, bit):
    max_val = torch.max(torch.abs(data))
    step = max_val / (2 ** (bit - 1) - 1)
    shift = torch.ceil(torch.log(step) / torch.log(torch.tensor([2.0])))

    scalarvar = torch.round(data / torch.pow(torch.tensor([2.0]), shift)).int()

    dquant_data = scalarvar * torch.pow(torch.tensor([2.0]), shift)
    return scalarvar, dquant_data.double(), shift.int()


def cnnl_quant_param(data, bit):
    tmp = torch.pow(torch.tensor([2.0]), bit-1)  - 1
    max_val = torch.max(torch.abs(data))
    if max_val== 0.0:
        pos = 0
        scale = 1.0
    elif bit != 31:
        pos = int(torch.floor(torch.log2(max_val))-(bit - 2))
        scale = torch.pow(torch.tensor([2.0]), pos)* tmp / max_val

    return pos, scale


data=torch.rand((3,4))

pos, scale=cnnl_quant_param(data, 8)
print(pos, scale)


t = data *  scale
d_q = torch.round(t/torch.pow(torch.tensor([2.0]), pos))

print(data)
print(d_q)

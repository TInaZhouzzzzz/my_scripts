import math
import torch
def get_quanz_param(data, bit):
    tmp = torch.pow(torch.tensor([2.0]), bit-1)  - 1
    max_val = torch.max(torch.abs(data))
    if max_val != 0.0:
        pos = int(torch.floor(torch.log2(max_val))-(bit - 2))
        scale = torch.pow(torch.tensor([2.0]), pos)* tmp / max_val
    return pos, scale

def quanz_data(x, pos, scale, bit):
    x_q=torch.clamp(torch.round(x*scale/(2**pos)),-2**(bit-1), 2**(bit-1)-1)    
    return x_q

def dequanz_data(x_q,  pos, scale, offset):
    x_deq= x_q * (2**pos) / scale
    return x_deq
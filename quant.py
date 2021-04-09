import torch
import math
def m_round(input, inplace=False):
    """ 模拟Round模式

    参数:
        input (torch.tensor): 待舍入的数据
        inplace (bool, default=False): 是否inplace舍入
        
    返回:
        input (torch.tensor): 舍入后的数据
    
    """
    try:
        round_mode ='to_even'
    except:
        round_mode == 'to_even'
    if round_mode == 'away_from_zero': 
        if inplace:
            input[:] = input.sign() * input.abs().add(0.5).floor()
        else:
            input = input.sign() * input.abs().add(0.5).floor()
    elif round_mode == 'to_even':
        if inplace:
            input[:] = input.round()
        else:
            input = input.round()
    else:
        raise ValueError

    return input


def _quanz_fix1(data, bit):
    max_val = torch.max(torch.abs(data))
    step = max_val / (2 ** (bit - 1) - 1)
    shift = torch.ceil(torch.log(step) / torch.log(torch.tensor([2.0])))

    scalarvar = torch.round(data / torch.pow(torch.tensor([2.0]), shift)).int()

    dquant_data = scalarvar * torch.pow(torch.tensor([2.0]), shift)
    return scalarvar, dquant_data.double(), shift.int()

def quanz_minmax_param(m_min, m_max, bit, is_scale, is_offset):
    r''' 根据最大值就散量化参数

    参数：
        m_min (torch.tensor)：原始数据的最小值
        m_max (torch.tensor)：原始数据的最大值
        bit (torch.tensor)：量化参数位宽
        is_scale (bool)： 量化是否使用scale
        is_offset (bool): 量化是否使用offset
    
    返回：
        m_position (torch.tensor)： 量化参数position
        m_scale (torch.tensor)：量化参数scale
        m_offset (torch.tensor)：量化参数offset

    '''
    device = m_max.device
    o_max = m_max.clone().detach()
    o_min = m_min.clone().detach()
    n = torch.tensor(bit).float().to(device)
    
    if is_offset:
        # 保证最大值不会小于0，保证最小值不会大于0
        m_max = torch.max(o_max, torch.zeros_like(o_max, dtype=torch.float32, device=device))
        m_min = torch.min(o_min, torch.zeros_like(o_min, dtype=torch.float32, device=device))
        interval = m_max - m_min
        # if interval is zero, the offset will inf, use 0.0 instead
        m_offset = torch.where(interval > 0.0,
                               m_round(-torch.pow(2.0, n - 1.0) - m_min * 2**1.0 * (torch.pow(2.0, n - 1.0) - 2**-1.0)/interval),
                               torch.zeros_like(interval, dtype=torch.float32, device=device))
        m_position = torch.where(interval > 0.0,
                                 torch.floor(torch.log2(interval)) - (n - 1.0),
                                 torch.zeros_like(interval, dtype=torch.float32, device=device))
        if is_scale:
            # if interval is zero, the scale will inf, use 0.0 instead
            m_scale = torch.where(interval > 0.0,
                                  (torch.pow(2.0, m_position+n) - torch.pow(2.0, m_position)) / interval,
                                  torch.ones_like(interval, dtype=torch.float32, device=device))
        else:
            m_scale = torch.ones_like(interval, dtype=torch.float32, device=device)
    else:
        # 取正负半轴的最大值
        m_max = torch.max(o_max, -o_min)
        m_offset = torch.zeros_like(m_max, dtype=torch.float32, device=device)
        m_position = torch.where(m_max > 0.0,
                                 torch.floor(torch.log2(m_max)) - (n-2.0),
                                 torch.zeros_like(m_max, dtype=torch.float32, device=device))
        if is_scale:
            # if interval is zero, the scale will inf, use 0.0 instead
            m_scale = torch.where(m_max > 0.0,
                                  (torch.pow(2.0, m_position+n-1.0) - torch.pow(2.0, m_position)) / m_max,
                                  torch.ones_like(m_max, dtype=torch.float32, device=device))
        else:
            m_scale = torch.ones_like(m_max, dtype=torch.float32, device=device)

    return m_position, m_scale, m_offset

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

def quant_param_offset(data, bit):
    tmp = torch.pow(torch.tensor([2.0]), bit-1)  - 1
    m_max = torch.max(torch.abs(data))
    m_min = torch.min(torch.abs(data))
    position, scale, offset = quanz_minmax_param( m_min, m_max, bit=8, is_scale=True, is_offset=True)
    return position, scale, offset


def  get_quanz_param(x, n):
    x_max=max(0, torch.max(x))
    x_min=min(0, torch.min(x))
    offset=-2**(n-1)
    pos=math.floor(math.log2(x_max-x_min))-(n-1)
    scale=(x_max-x_min)/((2**pos)*(2**n-1))
    return pos,scale,offset

def quanz_data(x, pos, scale, offset, n):
    x_q=torch.clamp(torch.round(x/(2**pos*scale))+offset,-2**(n-1),2**(n-1)-1)    
    return x_q

def dequanz_data(x_q,  pos, scale, offset):
    x_deq=(x_q-offset)*(2**pos*scale)
    return x_deq

for i in range(1):
    data=torch.rand((3,4))
    
    pos, scale=cnnl_quant_param(data, 8)
    print(pos, scale)
    pos2, scale2, o=quant_param_offset(data, 8)
    print(pos2, scale2, o)
    (pos,scale,offset)=get_quanz_param(data, 8)
    print("pos:",pos)
    print("scale:",scale)
    print("offset:",offset)
    
    
    t = data *  scale
    d_q = torch.round(t/torch.pow(torch.tensor([2.0]), pos))

    
    print("==================")
    print(data)
#   print(d_q)

    print("==================")
    x_q=quanz_data(data, pos, scale, offset, 8)
    print("x_q:",x_q)
    
    x_deq=dequanz_data(x_q,  pos, scale, offset)
    print("x_deq:",x_deq)
    print(scale*scale2)

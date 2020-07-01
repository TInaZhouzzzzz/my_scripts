from mxnet.gluon.model_zoo import vision
import mxnet as mx
import numpy as np
mobilenetv205 = vision.get_model('mobilenetv2_0.5',pretrained=True,classes=1000)
num_params = sum(np.prod(x.data().shape) for x in mobilenetv205.collect_params().values())
print(num_params)
x = mx.ndarray.random.uniform(shape=(1,3,224,224),ctx = mx.cpu())
print(mobilenetv205.summary(x))
mobilenetv205.hybridize()
mobilenetv205(x)
mobilenetv205.export('mobilenetv2')

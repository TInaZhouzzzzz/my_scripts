import mxnet as mx
model_path_0 = '/home/zhoushiyi/plot/'
net_name = 'mobilenetv2'
epoch =0
model_path = model_path_0 + net_name 
#model_path路径/home/fuxueping/sdb/PycharmProjects/mxnetTopytorch/se_best中保存的mxnet的网络结构和模型参数（.json,.param）
sym, arg_params, aux_params = mx.model.load_checkpoint(model_path, epoch)
#mx.viz.plot_network(sym, title='alexnet', save_format='jpg', hide_weights=True).view()
#mx.viz.plot_network(sym, title=net_name, save_format='jpg', hide_weights=True).view()
m = sym.list_arguments()
m_2 = sym.list_outputs()
m_9 = sym.get_internals().list_outputs()
m_3 = sym.get_internals()
m_4 = m_3.infer_shape(data=(1,3,224,224))

import numpy as np
import matplotlib.pyplot as plt
import sys
f_num = len(sys.argv)

if f_num == 1:
  print("Error: please give the log file path!!!")
  sys.exit()

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel('t_x_bit_pos')
plt.ylabel('grad_w_abs_max')
colors1 = '#00CED1' #点的颜色
colors2 = '#DC143C'
 
for i in range(f_num - 1):
    f_name = sys.argv[i+1]
    f1 = open(f_name)
    bit_pos = []
    grad_w_absmax = []
    b=[]
    g=[]
    ts=[]
    for j in f1:
        if "====" not in j:
#           print(j)
            bit_pos.append(float(j.strip('\n').split('\t')[1]))
            grad_w_absmax.append(float(j.strip('\n').split('\t')[3]))
            b.append(float(j.strip('\n').split('\t')[1]))
            g.append(float(j.strip('\n').split('\t')[3]))
        else:
            if len(b) > 0:
                m1 = max (b)
                m2=max(g)
                t = m2 / m1
                ts.append(t)
                b=[]
                g=[]
    print(ts)
    print(len(ts))
    plt.scatter(bit_pos, grad_w_absmax, alpha=0.4,  label=f_name.split('/')[-1])
  
#matplotlib画图中中文显示会有问题，需要这两行设置默认字体
 
#plt.xlim(xmax=9,xmin=0)
#plt.ylim(ymax=9,ymin=0)
 
#x1 = np.random.normal(2,1.2,300) # 随机产生300个平均值为2，方差为1.2的浮点数，即第一簇点的x轴坐标
#y1 = np.random.normal(2,1.2,300) # 随机产生300个平均值为2，方差为1.2的浮点数，即第一簇点的y轴坐标
#x2 = np.random.normal(7.5,1.2,300)
#y2 = np.random.normal(7.5,1.2,300)  
#area = np.pi * 4**2  # 点面积 
# 画散点图
#plt.scatter(x1, y1, c=colors1, alpha=0.4, label='resnet50')
#plt.scatter(x2, y2, c=colors2, alpha=0.4, label='vgg')
#plt.scatter(x1, y1, s=area, c=colors1, alpha=0.4, label='类别A')
#plt.scatter(x2, y2, s=area, c=colors2, alpha=0.4, label='类别B')
plt.plot([0,0],[2000,2000],linewidth = '0.5',color='#000000')
plt.legend()
plt.show()

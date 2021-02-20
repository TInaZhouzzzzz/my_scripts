import matplotlib.pyplot as plt
import sys
f_num = len(sys.argv)

if f_num == 1:
  print("Error: please give the log file path!!!")
  sys.exit()

L_dic={}
for i in range(f_num - 1):
  f_name = sys.argv[i+1]
  f1 = open(f_name)
  loss = []
  iters = []
  for j in f1:
    if "Loss" in j:
      loss.append(float(j.strip('\n').split(' ')[4]))
      iters.append(float(j.strip('\n').split(' ')[1]))
  print('get_data ending')
# print(iters)
# print(loss)
  plt.plot(iters,loss, label=f_name.split('/')[-1])
  plt.legend()

plt.show()
#     print(ws)
#     print(cs)
#     break

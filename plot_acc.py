import matplotlib.pyplot as plt
import sys
import re
f_num = len(sys.argv)

if f_num == 1:
  print("Error: please give the log file path!!!")
  sys.exit()
str_delete=[':', '[', ']', '(', ')', '/']
L_dic={}
for i in range(f_num - 1):
  f_name = sys.argv[i+1]
  f1 = open(f_name)
  loss = []
  iters = []
  for j in f1:
#   print(j)
    if "Test" in j:
      continue
    if "Loss" in j:
      for str1 in str_delete:
        j=j.replace(str1, ' ')
      j=re.sub(' +', ' ', j)
#     print(j)
#     print(j.split(' ')[11])
      loss.append(float(j.split(' ')[11]))
      iters.append(float(int(j.split(' ')[1])*5005 + int(j.split(' ')[2])))
# print('get_data ending')
# print(iters)
# print(loss)
  plt.plot(iters,loss, label=f_name.split('/')[-1])
  plt.legend()

plt.show()
#     print(ws)
#     print(cs)
#     break

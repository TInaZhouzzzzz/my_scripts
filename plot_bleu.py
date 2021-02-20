import matplotlib.pyplot as plt
import sys
f_num = len(sys.argv)

if f_num == 1:
  print("Error: please give the log file path!!!")
  sys.exit()


ws = {}
cs = {}

for i in range(f_num - 1):
  f_name = sys.argv[i+1]
  f1 = open(f_name)
  w = []
  c = []
  for j in f1:
    if "WER" in j:
      w.append(float(j.strip('\n').split(' ')[-3].split('\t')[0]))
      c.append(float(j.strip('\n').split(' ')[-1]))
  ws[f_name[-9:]]=w #.strip('/')[-1] 
  cs[f_name[-9:]]=c

plt.subplot(221)
l=[]
for k,v in ws.items():
  plt.plot(range(len(v)), v)
  l.append(k)
plt.ylabel("WER")
plt.xlabel("Epoch")
plt.legend(l,loc='upper right')

plt.subplot(222)
for k,v in cs.items():
  plt.plot(range(len(v)), v)
plt.ylabel("CER")
plt.xlabel("Epoch")
plt.legend(l,loc='upper right')

plt.show()
#     print(ws)
#     print(cs)
#     break



import matplotlib.pyplot as plt
import numpy as np

m, c, e = [],[],[]
f = open('result.txt','r')
for line in f:
    one = line.strip('\n')
    method,complexity,entropy = one.split(':')[0][1:-1],one.split(':')[1].split(',')[0][1:5],one.split(':')[1].split(',')[1][:4]
    m.append(method.capitalize())
#     c.append(complexity)
#     e.append(entropy)
    c.append(float(complexity))
    e.append(float(entropy))
f.close()

cnt = len(m)
# color_map = ['red','yellow','orange','goldenrod','pink','skyblue','cyan','steelblue']
color_map = ['#FF0000','#FFFF00','#FFA500','#DAA520','#FFC0CB','#87CEEB','#00FFFF','#4682B4']

e = np.array(e)
c = np.array(c)

for i in range(cnt): 
    plt.scatter(e[i], c[i], marker = 'o', color = color_map[i], s = 50, label = m[i]) 
    
plt.legend(loc = 'best')

plt.xlabel('Entropy,H')
plt.ylabel('Complexity,C')

plt.savefig('./CEplane.png',dpi = 600)
plt.show()

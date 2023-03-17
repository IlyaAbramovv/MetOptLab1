import matplotlib.pyplot as plt
import numpy as np

with open("data/ex6data_const_step.csv", 'r') as file:
    x, y, z = [], [], []
    while True:
        line = file.readline()
        if not line:
            break

        a, b, c, = map(float, line.split(","))
        x.append(a)
        y.append(b)
        z.append(c)


X, Y = np.meshgrid(x, y)
Z = np.meshgrid(y, z)[1]
fig, ax = plt.subplots()
c = ax.pcolormesh(X, Y, Z, cmap='coolwarm', shading='gouraud')
fig.colorbar(c, ax=ax)

ax.set_xlabel('n')
ax.set_ylabel('k')
plt.show()

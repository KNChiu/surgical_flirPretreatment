#%%
import matplotlib.pyplot as plt

# data x, y
x = range(35)
y = range(35)

# 點(x, y)的大小
z = range(35)

# colormap
cm = plt.cm.get_cmap('gnuplot2')

# generate figure
fig = plt.figure()

# 設定ax
ax = fig.add_subplot(1, 1, 1)

# 利用ax畫散佈圖
mappable = ax.scatter(x, y, c=z, vmin=0, vmax=35, s=35, cmap=cm)

# colorbar
fig.colorbar(mappable, ax=ax)

# show
plt.show()
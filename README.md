# 这是在跟mc讲了之后的版本 ，加上了 力随时间变化 2021.3.25 下午16.25



# 201.3.27添加了之前的底稿
这里面是学会远程库之前的版本
里面未改正 num 和 elm_nums 的错误
也为添加时间相关的力


 if i%2 == 0:
     plt.subplot(121)
     plt.plot(t,sol[:,0],label='i={}'.format(i) )
     plt.legend()
 if i%2 == 1:
     plt.subplot(122)
     plt.plot(t,sol[:,0],label='i={}'.format(i) )
     plt.legend()          # useless to plot it


未叠加回原坐标时的画图 删除！

# 添加力相关后 画图：mc要求的几个，以及整理代码
中点和端点的位移角度图
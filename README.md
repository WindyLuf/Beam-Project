# 目录说明
> 仿真课程相关
> > 仿真课程思维导图+仿真代码 类是注释版本
> 
> 平面三角单元
> > 弹性力学大作业 静力学有限元

> Beam FEA编程 底稿【未改正num版本】
> > test doc研一上搞得
> > > 最初的一些关于有限元的不成熟思想，此时还不明白模态分析。这里面的类文件也未注释
> 
> > beam 研一下搞得
> > > MC仿真课上FEA程序线性重写







## 这是在跟mc讲了之后的版本 ，加上了 力随时间变化 2021.3.25 下午16.25



## 201.3.27添加了之前的底稿
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

## 添加力相关后 画图：mc要求的几个，以及整理代码
> 中点和端点的位移角度图

## 2021.4.12 材料属性改为MC推荐版本
> 代码加入了自由振动 和 只有初始位移的版本
> 
> 添加了仿真课件进来
> 
> 修改了此MD文件

## 2021.5.6 存档部分代码，主要是记录公式
    if  2*np.pi*np.sqrt(M_i/K_i) > T_max:
        T_max = 2*np.pi*np.sqrt(M_i/K_i)      # Try to find the best time step
    if 2*np.pi*np.sqrt(M_i/K_i) < T_min:
        T_min = 2*np.pi*np.sqrt(M_i/K_i)
    if np.sqrt(K_i/M_i) < w_0:
        w_0 = np.sqrt(K_i/M_i)

print("\nMaximum   period :    T_0 =",T_max,"\nMinimum   period :    T_0 =",T_min,'\nMaximum angular speed:w_0 = ',w_0)
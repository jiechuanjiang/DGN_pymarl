This is the implementation of DGN on Pymarl, which could be trained by IQL, VDN or QMIX.

```
python3 src/main.py --config=vdn --env-config=sc2 with comm_flag=1.0
```

```
python3 src/main.py --config=iql --env-config=sc2 with comm_flag=1.0
```

In the costumed starcraft.py, we decrease the sight range (1) and communication range (5).


另，星际这个环境明显有问题，很多场景sight range设成0也能学，比如3s_vs_3z在sight range为0时用VDN能学出100%的胜率，原因在于智能体初始位置是基本固定的，obs里有智能体的id，rnn里隐含时间步，靠这两个信息就能直接过拟合最优解。

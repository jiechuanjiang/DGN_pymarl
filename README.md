This is the implementation of DGN on Pymarl, which could be trained by IQL, VDN or QMIX.

```
python3 src/main.py --config=vdn --env-config=sc2 with comm_flag=1.0
```

```
python3 src/main.py --config=iql --env-config=sc2 with comm_flag=1.0
```

In the costumed starcraft.py, we decrease the sight range (1) and communication range (5).

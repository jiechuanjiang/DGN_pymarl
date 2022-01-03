This is the implementation of DGN on Pymarl, which could be trained by VDN or QMIX.

```
python3 src/main.py --config=vdn --env-config=sc2 with comm_flag=1.0
```

```
python3 src/main.py --config=qmix --env-config=sc2 with comm_flag=1.0
```

In the costumed starcraft.py, we decrease the sight range and communication range.

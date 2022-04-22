# Word co-occurrence network transfer
We present a transformation method for word co-occurrence networks that can achieve better word representations.

## requirements
python==3.8  
torch==1.10  
scikit-learn==0.24.0  
networkx==2.6.3  
numpy==1.20.3  
tqdm  

## Network transfer
Network transfer can be conducted by:
```shell
python3 1.network_make.py yes
```
If you need original co-occurrence network, command is changed to:
```shell
python3 1.network_make.py no
```

## Network embedding
here, we use LINE for network embedding, and command for embedding is:
```shell
python3 2.network_embedding.py yes
```
there still is an origin version:
```shell
python3 2.network_embedding.py no
```

## Add: Classify
if you have labels for your nodes, you can do classify test for your embedding result. replace `data-for-test/label-list.json` by your labels infer, and conduct:
```shell
python3 3.classify.py
```

Good luck!
import pandas as pd
from PAMI.correlatedPattern.basic.CoMinePlus import CoMinePlus as alg
import warnings

warnings.filterwarnings("ignore")

# CoMine algorithm from PAMI
def test_pami(dataset, min_sup=0.2, min_all_conf=0.2):
    dataset = [",".join(i) for i in dataset]
    with open("sample.csv", "w+") as f:
        f.write("\n".join(dataset))
    obj = alg(iFile="sample.csv", minSup=min_sup, minAllConf=min_all_conf, sep=',')
    obj.mine()
    res = obj.getPatternsAsDataFrame()
    res["Patterns"] = res["Patterns"].apply(lambda x: x.split())
    res["Support"] = res["Support"].apply(lambda x: x / len(dataset))
    pami = res
    return pami


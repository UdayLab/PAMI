import pandas as pd
from gen import generate_transactional_dataset
from PAMI.recurringPattern.basic import RPGrowth as alg
import warnings

warnings.filterwarnings("ignore")

# RPGrowth algorithm from PAMI
def test_pami(dataset, min_sup=0.2, max_period_count=5000, min_rec=1.8):
    dataset = [",".join(map(str, i)) for i in dataset]
    with open("sample.csv", "w+") as f:
        f.write("\n".join(dataset))
    obj = alg.RPGrowth(iFile="sample.csv", minPS=min_sup, maxPer=max_period_count, minRec=min_rec, sep=',')
    obj.startMine()  # Using mine() instead of the deprecated startMine()
    res = obj.getPatternsAsDataFrame()
    res["Patterns"] = res["Patterns"].apply(lambda x: x.split())
    res["Support"] = res["Support"].apply(lambda x: x / len(dataset))
    pami = res
    return pami

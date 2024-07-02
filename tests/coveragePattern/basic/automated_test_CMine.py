import pandas as pd
from gen import generate_transactional_dataset
from PAMI.coveragePattern.basic.CMine import CMine as alg
import warnings

warnings.filterwarnings("ignore")

# CMine algorithm from PAMI
def test_pami(dataset, min_rf=0.0006, min_cs=0.3, max_or=0.5):
    dataset = [",".join(i) for i in dataset]
    with open("sample.csv", "w+") as f:
        f.write("\n".join(dataset))
    obj = alg(iFile="sample.csv", minRF=min_rf, minCS=min_cs, maxOR=max_or, sep=',')
    obj.mine()
    res = obj.getPatternsAsDataFrame()
    res["Patterns"] = res["Patterns"].apply(lambda x: x.split())
    res["Support"] = res["Support"].apply(lambda x: x / len(dataset))
    return res

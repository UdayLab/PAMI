import pandas as pd
from gen import generate_transactional_dataset
from PAMI.coveragePattern.basic.CPPG import CPPG as alg
import warnings

warnings.filterwarnings("ignore")

# CPPG algorithm from PAMI
def test_pami(dataset, min_rf=0.0006, min_cs=0.3, max_or=0.5):
    dataset = [",".join(i) for i in dataset]
    with open("sample_cppg.csv", "w+") as f:
        f.write("\n".join(dataset))
    obj = alg(iFile="sample_cppg.csv", minRF=min_rf, minCS=min_cs, maxOR=max_or, sep=',')
    obj.mine()
    res = obj.getPatternsAsDataFrame()
    res["Patterns"] = res["Patterns"].apply(lambda x: x.split())
    # Assuming the support calculation is similar to Apriori's, adjust as necessary
    res["Support"] = res["Support"].apply(lambda x: x / len(dataset))
    cppg = res
    return cppg

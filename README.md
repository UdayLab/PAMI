![PyPI](https://img.shields.io/pypi/v/PAMI)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/PAMI)
[![GitHub license](https://img.shields.io/github/license/UdayLab/PAMI)](https://github.com/UdayLab/PAMI/blob/main/LICENSE)
![PyPI - Implementation](https://img.shields.io/pypi/implementation/PAMI)
[![Documentation Status](https://readthedocs.org/projects/pami-1/badge/?version=latest)](https://pami-1.readthedocs.io/en/latest/?badge=latest)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/PAMI)
![PyPI - Status](https://img.shields.io/pypi/status/PAMI)
[![GitHub issues](https://img.shields.io/github/issues/UdayLab/PAMI)](https://github.com/UdayLab/PAMI/issues)
[![GitHub forks](https://img.shields.io/github/forks/UdayLab/PAMI)](https://github.com/UdayLab/PAMI/network)
[![GitHub stars](https://img.shields.io/github/stars/UdayLab/PAMI)](https://github.com/UdayLab/PAMI/stargazers)
[![Downloads](https://static.pepy.tech/badge/pami)](https://pepy.tech/project/pami)
[![Downloads](https://static.pepy.tech/badge/pami/month)](https://pepy.tech/project/pami)
[![Downloads](https://static.pepy.tech/badge/pami/week)](https://pepy.tech/project/pami)

[Click here for more information](https://pepy.tech/project/pami)


# Introduction
PAttern MIning (PAMI) is a Python library containing several algorithms to discover user interest-based patterns in a wide-spectrum of datasets across multiple computing platforms. Useful links to utilize the services of this library were provided below:


1. User manual https://udaylab.github.io/PAMI/manuals/index.html

2. Coders manual https://udaylab.github.io/PAMI/codersManual/index.html

3. Code documentation https://pami-1.readthedocs.io 

4. Datasets   https://u-aizu.ac.jp/~udayrage/datasets.html

5. Discussions on PAMI usage https://github.com/UdayLab/PAMI/discussions

6. Report issues https://github.com/UdayLab/PAMI/issues

# Features

- ✅ Well-tested and production ready
- 🔋 High optimized to our best-effort, light-weight, and energy efficient
- 👀 Proper code documentation
- 🍼 Ample examples on using various algorithms at [./notebooks](https://github.com/UdayLab/PAMI/tree/main/notebooks) folder
- 🤖 Works with AI libraries such as TensorFlow, PyTorch, and sklearn. 
- ⚡️ Supports Cuda and PySpark 
- 🖥️ Operating System Independence
- 🔬 Knowledge discovery in static data and streams
- 🐎 Snappy
- 🐻 Ease of use

# Recent versions  

- Version 2023.07.07: New algorithms: cuApriroi, cuAprioriBit, cuEclat, cuEclatBit, gPPMiner, cuGPFMiner, FPStream, HUPMS, SHUPGrowth New codes to generate synthetic databases
- Version 2023.06.20: Fuzzy Partial Periodic, Periodic Patterns in High Utility, Code Documentation, help() function Update 
- Version 2023.03.01: prefixSpan and SPADE   

Total number of algorithms: 83

# Maintenance

  Installation
  
       pip install pami
       pip install 'pami[gpu]'
       pip install 'pami[spark]'
  
  Updation
  
       pip install --upgrade pami
  
  Uninstallation
  
       pip uninstall pami 
       
# Tutorials 

## 1. Mining Databases
### Transactional databases
1. Frequent pattern mining: [Sample](https://udayrage.github.io/PAMI/frequentPatternMining.html)

| Basic                                                                                                                                                                                                                                                     | Closed                                                                                                                                                                                                                   | Maximal                                                                                                                             | Top-k                                                                                                                                                                                                | CUDA           | pyspark                                                                                                                                                                                                                                                                             |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Apriori <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/Apriori.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                   | Closed [Basic](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/frequentPattern/closed/CHARM/CHARM-st.md)-[Adv](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/frequentPattern/closed/CHARM/CHARM-ad.md) | maxFP-growth [Basic](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/frequentPattern/maximal/MaxFPGrowth/MaxFPGrowth-st.md) | topK [Basic](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/frequentPattern/topk/FAE-st.pdf)-[Adv](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/frequentPattern/topk/FAE-ad.pdf) | cudaAprioriGCT | parallelApriori [Basic](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/frequentPattern/pyspark/parallelApriori/parallelApriori-st.md)-[Adv](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/frequentPattern/pyspark/parallelApriori/ParallelApriori-ad%20(1).md)   |
| FP-growth <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/FPGrowth.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                |                                                                                                                                                                                                                          |                                                                                                                                     |                                                                                                                                                                                                      | cudaAprioriTID | parallelFPGrowth [Basic](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/frequentPattern/pyspark/parallelFPGrowth/parallelFPGrowth-st.md)-[Adv](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/frequentPattern/pyspark/parallelFPGrowth/ParallelFPGrowth-ad%20.md) |
| ECLAT  <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/FPGrowth.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                   |                                                                                                                                                                                                                          |                                                                                                                                     |                                                                                                                                                                                                      | cudaEclatGCT   | parallelECLAT [Basic](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/frequentPattern/pyspark/parallelECLAT/parallelECLAT-st.md)-[Adv](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/frequentPattern/pyspark/ParallelECLAT-ad.pdf)                                |
| ECLAT-bitSet [Basic](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/frequentPattern/basic/ECLATbitset/ECLATbitset-st.md)-[Adv](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/frequentPattern/basic/ECLATbitset/Eclatbitset-ad.md)      |                                                                                                                                                                                                                          |                                                                                                                                     |                                                                                                                                                                                                      |                |                                                                                                                                                                                                                                                                                     |
| ECLAT-diffset [Basic](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/frequentPattern/basic/ECLATDiffset/ECLATDiffset-st.md)-[Adv](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/frequentPattern/basic/ECLATDiffset/EclatDiffset-ad.md) |                                                                                                                                                                                                                          |                                                                                                                                     |                                                                                                                                                                                                      |                |

2. Relative frequent pattern mining: [Sample](https://udayrage.github.io/PAMI/relativeFrequentPatternMining.html)

| Basic                                                                                                                                                                                                                                          |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| RSFP  [Basic](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/relativeFrequentPatterns/RSFPGrowth/RSFPGrowth-st.md)-[Adv](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/relativeFrequentPatterns/RSFPGrowth/RSFPGrowt-ad.md) |


3. Frequent pattern with multiple minimum support: [Sample](https://udayrage.github.io/PAMI/multipleMinSupFrequentPatternMining.html)

| Basic       |
|-------------|
| CFPGrowth   |
| CFPGrowth++ |



4. Correlated pattern mining: [Sample](https://udayrage.github.io/PAMI/correlatePatternMining.html)

| Basic                                                                                                                                                                                                                                                 |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| CoMine <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/correlatedPattern/basic/CoMine.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>       |
| CoMine++ <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/correlatedPattern/basic/CoMinePuls.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |

5. Fault-tolerant frequent pattern mining (under development)

| Basic                                                                                                                                                                                                                                                             |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| FTApriori <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/faultTolerantFrequentPatterns/basic/FTApriori.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |
| FTFPGrowth (under development)                                                                                                                                                                                                                                    |

6. Coverage pattern mining (under development)

| Basic                                                                                                                                                                                                                                             |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| CMine <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/coveragePattern/basic/CMine.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>       |
| CMine++ <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/coveragePattern/basic/CMinePlus.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |

### Temporal databases


1. Periodic-frequent pattern mining: [Sample](https://udayrage.github.io/PAMI/periodicFrequentPatternMining.html)

| Basic                                                                                                                                                                                                                                                              | Closed                                                                                                                                                                                                                                                  | Maximal                                                                                                                                                                                                                                                           | Top-K                                                                                                                                                                                                                                                      |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| PFP-growth <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/periodicFrequentPattern/basic/PFPGrowth.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>       | CPFP <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/periodicFrequentPattern/closed/CPFPMiner.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> | maxPF-growth<a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/periodicFrequentPattern/maximal/MaxPFGrowth.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> | kPFPMiner <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/periodicFrequentPattern/topk/kPFPMiner.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |
| PFP-growth++ <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/periodicFrequentPattern/basic/PFPGrowthPlus.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |                                                                                                                                                                                                                                                         | Topk-PFP<a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/periodicFrequentPattern/topk/TopKPFP.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>            |
| PS-growth <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/periodicFrequentPattern/basic/PSGrowth.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>         |                                                                                                                                                                                                                                                         |                                                                                                                                                                                                                                                                   |
| PFP-ECLAT <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/periodicFrequentPattern/basic/PFECLAT.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>           |                                                                                                                                                                                                                                                         |                                                                                                                                                                                                                                                                   |
 | PFPM-Compliments <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/periodicFrequentPattern/basic/PFPMC.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>     |                                                                                                                                                                                                                                                         |                                                                                                                                                                                                                                                                   |

2. Local periodic pattern mining: [Sample](https://udayrage.github.io/PAMI/localPeriodicPatternMining.html)

| Basic                                                                                                                                  |
|----------------------------------------------------------------------------------------------------------------------------------------|
| LPPGrowth   [Basic](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/localPeriodicPatterns/basic/LPPGrowth/LPPGrowth-st.md)     |
| LPPMBreadth [Basic](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/localPeriodicPatterns/basic/LPPMBreadth/LPPMBreadth-st.md) |
| LPPMDepth   [Basic](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/localPeriodicPatterns/basic/LPPMDepth/LPPMDepth-st.md)     |

3. Partial periodic-frequent pattern mining: [Sample](https://udayrage.github.io/PAMI/partialPeriodicFrequentPattern.html)

| Basic                                                                                                                                                                                                                                              |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| GPF-growth [Basic](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/partialPeriodicFrequentPatterns/GPFGrowth/GPFgrowth-st.md)-[Adv](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/partialPeriodicFrequentPattern/PPF_DFS-ad.pdf) |
| PPF-DFS [Basic](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/partialPeriodicFrequentPatterns/PPF_DFS/PPF_DFS-st.md)-[Adv](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/partialPeriodicFrequentPattern/PPF_DFS-ad.pdf)        |

4. Partial periodic pattern mining: [Sample](https://udayrage.github.io/PAMI/partialPeriodicPatternMining.html)

| Basic                                                                                                                                                                                                                                                              | Closed                                                                                                                                                                                                                               | Maximal                                                                                                                         |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| 3P-growth [Basic](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/partialPeriodicPatterns/basic/PPPGrowth/PPPGrowth-st.md)-[Adv](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/partialPeriodicPatterns/basic/PPPGrowth/PPPGrowth-ad.md)          | 3P-close [Basic](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/partialPeriodicPattern/closed/PPPClose-st.pdf)-[Adv](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/partialPeriodicPattern/closed/PPPClose-ad.pdf) | max3P-growth [Basic](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/partialPeriodicPattern/maximal/Max3PGrowth-st.pdf) |                                                                                                                                                                                                                                         |                                                                                                                                  |               | |
| 3P-ECLAT [Basic](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/partialPeriodicPatterns/basic/PPPECLAT/PPP_ECLAT-st.md)-[Adv](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/partialPeriodicPatterns/basic/PPPECLAT/PPP_ECLAT-ad.md)             |                                                                                                                                                                                                                                      |                                                                                                                                 |
| G3P-Growth [Basic](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/partialPeriodicPatterns/basic/G3PGrowth/GThreePGrowth-st.md)-[Adv](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/partialPeriodicPatterns/basic/G3PGrowth/GThreePGrowth-ad.md) |                                                                                                                                                                                                                                      |                                                                                                                                 |


5. Periodic correlated pattern mining: [Sample](https://udayrage.github.io/PAMI/periodicCorrelatedPatternMining.html)

| Basic                                                                                                                                                                                                                                                                     |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| EPCP-growth [Basic](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/partialPeriodicPatterns/correlated/EPCPGrowth/EPCPGrowth-st.md)-[Adv](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/partialPeriodicPatterns/correlated/EPCPGrowth/EPCPGrowth-ad.md) |

6. Stable periodic pattern mining: [Sample](https://udayrage.github.io/PAMI/stablePeriodicPatterns.html)

| Basic                                                                                                                                                                                                                                                           | TopK  |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------|
| SPP-growth <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/stablePeriodicPatterns/basic/SPPGrowth.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> | TSPIN <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/stablePeriodicPatterns/topk/TSPIN.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>|
| SPP-ECLAT <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/stablePeriodicPatterns/basic/SPPECLAT.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>   |       |


### Geo-referenced (or spatiotemporal) databases

1. Frequent spatial pattern mining: [Sample](https://udayrage.github.io/PAMI/frequentSpatialPatternMining.html)

| Basic                                                                                                                                                                                                                               |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| spatialECLAT  [Basic](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/frequentSpatialPattern/SpatialECLAT-st.pdf)-[Adv](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/frequentSpatialPattern/SpatialECLAT-ad.pdf) |
| FSP-growth [Basic](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/frequentSpatialPattern/FSPGrowth-st.pdf)-[Adv](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/frequentSpatialPattern/FSPGrowth-ad.pdf)          |

2. Geo referenced Periodic frequent pattern mining: [Sample](https://udayrage.github.io/PAMI/periodicFrequentSpatial.html)

| Basic                                                                                                                                                                                                                                                           |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| GPFPMiner [Basic](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/periodicFrequentPatterns/spatial/GPFPMiner/GPFPMiner-st.md)-[Adv](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/periodicFrequentPatterns/spatial/GPFPMiner/GPFPMiner-ad.md) |

3. Partial periodic spatial pattern mining:[Sample](https://udayrage.github.io/PAMI/partialPeriodicSpatialPatternMining.html)

| Basic                                                                                                                                                                                                                                               |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| STECLAT [Basic](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/partialPeriodicPatterns/spatial/STECLAT/STECLAT-st.md)-[Adv](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/partialPeriodicPatterns/spatial/STECLAT/STECLAT-ad.md) |

4. Recurring pattern mining: [Sample](https://udayrage.github.io/PAMI/RecurringPatterns.html)

| Basic    |
|----------|
| RPgrowth <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/recurringPatterns/basic/RPGrowth.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>|


### Utility databases

1. High utility pattern mining:   [Sample](https://udayrage.github.io/PAMI/highUtilityPatternMining.html)

| Basic    |
|----------|
| EFIM  [Basic](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/highUtilityPatterns/basic/EFIM/EFIM-st.md)-[Adv](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/highUtilityPatterns/basic/EFIM/EFIM-ad.md)   |
| HMiner [Basic](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/highUtilityPatterns/basic/EFIM/EFIM-st.md)-[Adv](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/highUtilityPatterns/basic/HMiner/HMiner-st.md)  |
| UPGrowth |

2. High utility frequent pattern mining:  [Sample](https://udayrage.github.io/PAMI/highUtiltiyFrequentPatternMining.html)

| Basic |
|-------|
| HUFIM [Basic](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/hitghUtilityFrequent/basic/HUFIM/HUFIM-st-2.md)-[Adv](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/hitghUtilityFrequent/basic/HUFIM/HUFIM-ad.md)|


3. High utility frequent spatial pattern mining:  [Sample](https://udayrage.github.io/PAMI/highUtilitySpatialPatternMining.html)

| Basic  |
|--------|
| SHUFIM [Basic](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/hitghUtilityFrequent/spatial/SHUFIM-st%20.md)-[Adv](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/hitghUtilityFrequent/spatial/SHUFIM-ad.md)|

4. High utility spatial pattern mining:  [Sample](https://udayrage.github.io/PAMI/highUtilitySpatialPatternMining.html)

| Basic  | topk    |
|--------|---------|
| HDSHIM [Basic](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/highUtilityPatterns/spatial/HDSHUIM/HDSHUIM-st%20-2.md)-[Adv](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/highUtilityPatterns/spatial/HDSHUIM/HDSHUIM-ad.md)| TKSHUIM |
| SHUIM  [Basic](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/highUtilityPatterns/spatial/SHUM/SHUIM-st.md)|


5. Relative High utility pattern mining: [Sample](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/mainManuals/relativeUtility.html)

| Basic |
|-------|
| RHUIM [Basic](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/highUtilityPatterns/relative/RHUIM-st.md)-[Adv](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/highUtilityPatterns/relative/RHUIM-ad.md)| 


6. Weighted frequent pattern mining: [Sample](https://udayrage.github.io/PAMI/weightedFrequentPattern.html)

| Basic                                                                                                                                                                                                                                               |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| WFIM  <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/weightedFrequentPatterns/basic/WFIM.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |


7. Weighted frequent regular pattern mining: [Sample](https://udayrage.github.io/PAMI/weightedFrequentRegularPatterns.html)

| Basic                                                                                                                                                                                                                                                          |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| WFRIMiner <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/weightedFrequentRegularPatterns/basic/WFRI.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |


8. Weighted frequent neighbourhood pattern mining: [Sample](https://github.com/UdayLab/PAMI/blob/main/docs/weightedSpatialFrequentPattern.html)

| Basic       |
|-------------|
| SSWFPGrowth |

### Fuzzy databases
1. Fuzzy Frequent pattern mining: [Sample](https://github.com/UdayLab/PAMI/fuzzyFrequentPatternMining.html)

| Basic                                                                                                                                                                                                                   |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| FFI-Miner [Basic](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/fuzzyFrequentPatterns/basic/FFIMiner/FFIMiner-st.md)-[Adv](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/fuzzyFrequentPatterns/FFIMiner-ad.pdf) |




2. Fuzzy correlated pattern mining: [Sample](https://udayrage.github.io/PAMI/fuzzyCorrelatedPatternMining.html)

| Basic                                                                                                                                                                                                                        |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| FCP-growth [Basic](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/fuzzyCorrelatedPatterns/basic/FCPGrowth/FCPGrowth-st%20.md)-[Adv](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/fuzzyCorrelatedPatterns/basic/FCPGrowth/FCPGrowth-ad.md) |


3. Fuzzy frequent spatial pattern mining: [Sample](https://github.com/UdayLab/PAMI/fuzzyFrequentSpatialPatternMining.html)

| Basic                                                                                                                                                                                                                                  |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| FFSP-Miner [Basic](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/fuzzyFrequentPatterns/fuzzySpatial/FFSPMiner/FFSPMiner-ad.md)-[Adv](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/fuzzyFrequentPatterns/fuzzySpatial/FFSPMiner/FFSPMiner-st.md) |

4. Fuzzy periodic frequent pattern mining: [Sample](https://github.com/UdayLab/PAMI/fuzzyPeriodicFrequentPatternMining.html)

| Basic                                                                                                                                                                                                                                    |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| FPFP-Miner [Basic](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/fuzzyPeriodicFrequentPatterns/basic/FPFPMiner/FPFPMiner-st.md)-[Adv](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/fuzzyPeriodicFrequentPatterns/basic/FPFPMiner/FPFPMiner-ad.md) |

5. Geo referenced Fuzzy periodic frequent pattern mining: [Sample](https://github.com/UdayLab/PAMI/fuzzySpatialPeriodicFrequentPatternMining.html)

| Basic                                                                                                                                                                                                                                    |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| FPFP-Miner [Basic](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/fuzzyPeriodicFrequentPattern/FPFPMiner-st.pdf)-[Adv](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/fuzzyPeriodicFrequentPattern/FPFPMiner-ad.pdf) |

### Uncertain databases


1. Uncertain frequent pattern mining: [Sample](https://udayrage.github.io/PAMI/uncertainFrequentPatternMining.html)

| Basic                                                                                                                                                                                                                                                   | top-k |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------|
| PUF <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/uncertainFrequentPatterns/basic/PUFGrowth.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> | TUFP  |
| TubeP  <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/uncertainFrequentPatterns/basic/TubeP.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>  |       |
| TubeS  <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/uncertainFrequentPatterns/basic/TubeS.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>  |       | 
| UVEclat                                                                                                                                                                                                                                                 |       |

2. Uncertain periodic frequent pattern mining: [Sample](https://udayrage.github.io/PAMI/uncertainPeriodicFrequentPatternMining.html)

| Basic                                                                                                                                                                                                                                                                          |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| UPFP-growth <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/uncertainPeriodicFrequentPatterns/basic/UPFPGrowth.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>       |
| UPFP-growth++ <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/uncertainPeriodicFrequentPatterns/basic/UPFPGrowthPlus.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |


3. Uncertain Weighted frequent pattern mining: [Sample](https://udayrage.github.io/PAMI/weightedUncertainFrequentPatterns.html)

| Basic                                                                                                                                                                                                                                                 |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| WUFIM <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/uncertainWeightedFrequent/basic/WUFIM.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |

### Sequence databases

1. Sequence frequent pattern mining: [Sample](https://github.com/UdayLab/PAMI/blob/main/docs/weightedSpatialFrequentPattern.html)
    
| Basic                                                                                                                                                                                                                                                       |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| SPADE <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/sequencePatternMining/basic/Spade.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>           |
| PrefixSpan <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/sequencePatternMining/basic/PrifixSpan.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |


2. Geo-referenced Frequent Sequence Pattern mining

| Basic |
|-------|
|GFSP-Miner|

### Timeseries databases


## 2. Mining Streams

1. Frequent pattern mining

|Basic|
|-----|
 |to be written|

2. High utility pattern mining

| Basic |
|-------|
 | HUPMS |



## 3. Mining Graphs
__coming soon__   
     
     

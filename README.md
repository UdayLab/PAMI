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


1. Youtube tutorial https://www.youtube.com/playlist?list=PLKP768gjVJmDer6MajaLbwtfC9ULVuaCZ

2. User manual https://udaylab.github.io/PAMI/manuals/index.html

3. Coders manual https://udaylab.github.io/PAMI/codersManual/index.html

4. Code documentation https://pami-1.readthedocs.io 

5. Datasets   https://u-aizu.ac.jp/~udayrage/datasets.html

6. Discussions on PAMI usage https://github.com/UdayLab/PAMI/discussions

7. Report issues https://github.com/UdayLab/PAMI/issues

# Features

- ✅ Well-tested and production-ready
- 🔋 Highly optimized to our best effort, light-weight, and energy efficient
- 👀 Proper code documentation
- 🍼 Ample examples of using various algorithms at [./notebooks](https://github.com/UdayLab/PAMI/tree/main/notebooks) folder
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

  __Installation__
  
  1. Installing basic pami package (recommended)


         pip install pami


  2. Installing pami package in a GPU machine that supports CUDA


         pip install 'pami[gpu]'


  3. Installing pami package in a distributed network environment supporting Spark


         pip install 'pami[spark]'
  

  __Upgradation__

  
        pip install --upgrade pami
  

  __Uninstallation__

  
        pip uninstall pami 
       

  __Information__ 


        pip show pami


***

# Tutorials 

### 1. Pattern mining in binary transactional databases

#### 1.1. Frequent pattern mining: [Sample](https://udaylab.github.io/PAMI/frequentPatternMining.html)

| Basic                                                                                                                                                                                                                                                      | Closed                                                                                                                                                                                                                                       | Maximal                                                                                                                                                                                                                                                     | Top-k                                                                                                                                                                                                                                  | CUDA           | pyspark                                                                                                                                                                                                                                                             |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Apriori <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/frequentPattern/basic/Apriori.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>              | CHARM <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/frequentPattern/closed/CHARM.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> | maxFP-growth  <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/frequentPattern/maximal/MaxFPGrowth.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> | FAE <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/frequentPattern/topk/FAE.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> | cudaAprioriGCT | parallelApriori <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/frequentPattern/pyspark/parallelApriori.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>   |
| FP-growth <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/frequentPattern/basic/FPGrowth.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>           |                                                                                                                                                                                                                                              |                                                                                                                                                                                                                                                             |                                                                                                                                                                                                                                        | cudaAprioriTID | parallelFPGrowth <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/frequentPattern/pyspark/parallelFPGrowth.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |
| ECLAT  <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/frequentPattern/basic/ECLAT.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                 |                                                                                                                                                                                                                                              |                                                                                                                                                                                                                                                             |                                                                                                                                                                                                                                        | cudaEclatGCT   | parallelECLAT <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/frequentPattern/pyspark/parallelECLAT.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>       |
| ECLAT-bitSet <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/frequentPattern/basic/ECLATbitset.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>   |                                                                                                                                                                                                                                              |                                                                                                                                                                                                                                                             |                                                                                                                                                                                                                                        |                |                                                                                                                                                                                                                                                                     |
| ECLAT-diffset <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/frequentPattern/basic/ECLATDiffset.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |                                                                                                                                                                                                                                              |                                                                                                                                                                                                                                                             |                                                                                                                                                                                                                                        |                |

#### 1.2. Relative frequent pattern mining: [Sample](https://udaylab.github.io/PAMI/relativeFrequentPatternMining.html)

| Basic                                                                                                                                                                                                                                                          |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| RSFP-growth <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/relativeFrequentPattern/basic/RSFPGrowth.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |


#### 1.3. Frequent pattern with multiple minimum support: [Sample](https://udaylab.github.io/PAMI/multipleMinSupFrequentPatternMining.html)

| Basic                                                                                                                                                                                                                                                                     |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| CFPGrowth <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/multipleMinimumFrequentPatterns/basic/CFPGrowth.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>       |
| CFPGrowth++ <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/multipleMinimumFrequentPatterns/basic/CFPGrowthPlus.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |


#### 1.4. Correlated pattern mining: [Sample](https://udaylab.github.io/PAMI/correlatePatternMining.html)

| Basic                                                                                                                                                                                                                                                 |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| CoMine <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/correlatedPattern/basic/CoMine.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>       |
| CoMine++ <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/correlatedPattern/basic/CoMinePuls.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |

#### 1.5. Fault-tolerant frequent pattern mining (under development)

| Basic                                                                                                                                                                                                                                                                                   |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| FTApriori <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/faultTolerantFrequentPatterns/basic/FTApriori.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>                       |
| FTFPGrowth (under development) <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/faultTolerantFrequentPatterns/basic/FTFPGrowth.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |

#### 1.6. Coverage pattern mining (under development)

| Basic                                                                                                                                                                                                                                             |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| CMine <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/coveragePattern/basic/CMine.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>       |
| CMine++ <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/coveragePattern/basic/CMinePlus.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |

### 2. Pattern mining in binary temporal databases

#### 2.1. Periodic-frequent pattern mining: [Sample](https://udaylab.github.io/PAMI/periodicFrequentPatternMining.html)

| Basic                                                                                                                                                                                                                                                              | Closed                                                                                                                                                                                                                                                  | Maximal                                                                                                                                                                                                                                                           | Top-K                                                                                                                                                                                                                                                      |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| PFP-growth <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/periodicFrequentPattern/basic/PFPGrowth.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>       | CPFP <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/periodicFrequentPattern/closed/CPFPMiner.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> | maxPF-growth<a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/periodicFrequentPattern/maximal/MaxPFGrowth.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> | kPFPMiner <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/periodicFrequentPattern/topk/kPFPMiner.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |
| PFP-growth++ <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/periodicFrequentPattern/basic/PFPGrowthPlus.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |                                                                                                                                                                                                                                                         | Topk-PFP<a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/periodicFrequentPattern/topk/TopKPFP.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>            |
| PS-growth <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/periodicFrequentPattern/basic/PSGrowth.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>         |                                                                                                                                                                                                                                                         |                                                                                                                                                                                                                                                                   |
| PFP-ECLAT <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/periodicFrequentPattern/basic/PFECLAT.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>           |                                                                                                                                                                                                                                                         |                                                                                                                                                                                                                                                                   |
 | PFPM-Compliments <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/periodicFrequentPattern/basic/PFPMC.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>     |                                                                                                                                                                                                                                                         |                                                                                                                                                                                                                                                                   |

#### 2.2. Local periodic pattern mining: [Sample](https://udaylab.github.io/PAMI/localPeriodicPatternMining.html)

| Basic                                                                                                                                                                                                                                                                             |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| LPPGrowth  (under development) <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/localPeriodicPattern/basic/LPPgrowth.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>     |
| LPPMBreadth (under development) <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/localPeriodicPattern/basic/LPPM_breadth.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |
| LPPMDepth   (under development) <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/localPeriodicPattern/basic/LPPM_depth.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>   |

#### 2.3. Partial periodic-frequent pattern mining: [Sample](https://udaylab.github.io/PAMI/partialPeriodicFrequentPattern.html)

| Basic                                                                                                                                                                                                                                                               |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| GPF-growth <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/partialPeriodicFrequentPattern/basic/GPFgrowth.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |
| PPF-DFS <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/partialPeriodicFrequentPattern/basic/PPF_DFS.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>      |
| GPPF-DFS <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/partialPeriodicFrequentPattern/basic/GPFgrowth.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>      |


#### 2.4. Partial periodic pattern mining: [Sample](https://udaylab.github.io/PAMI/partialPeriodicPatternMining.html)

| Basic                                                                                                                                                                                                                                                           | Closed                                                                                                                                                                                                                                                    | Maximal                                                                                                                                                                                                                                                           | topK                                                                                                                                                                                                                                                                | CUDA                                                                                                                                                                                                                                                                            |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 3P-growth <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/partialPeriodicPattern/basic/PPPGrowth.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>      | 3P-close <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/partialPeriodicPattern/closed/PPPClose.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> | max3P-growth <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/partialPeriodicPattern/maximal/Max3PGrowth.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> | topK-3P growth <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/partialPeriodicPattern/topk/Topk_PPPGrowth.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> | cuGPPMiner (under development) <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/partialPeriodicPattern/cuda/cuGPPMiner.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |                                                                                                                                                                                                                                  |                                                                                                                                  |               | |
| 3P-ECLAT <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/partialPeriodicPattern/basic/PPP_ECLAT.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>       |                                                                                                                                                                                                                                                           |                                                                                                                                                                                                                                                                   |                                                                                                                                                                                                                                                                     | gPPMiner (under development)  <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/partialPeriodicPattern/cuda/gPPMiner.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>    |
| G3P-Growth <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/partialPeriodicPattern/basic/GThreePGrowth.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |                                                                                                                                                                                                                                                           |                                                                                                                                                                                                                                                                   |                                                                                                                                                                                                                                                                     |                                                                                                                                                                                                                                                                                 |


#### 2.5. Periodic correlated pattern mining: [Sample](https://udaylab.github.io/PAMI/periodicCorrelatedPatternMining.html)

| Basic                                                                                                                                                                                                                                                            |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| EPCP-growth <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/periodicCorrelatedPattern/basic/EPCPGrowth.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |

#### 2.6. Stable periodic pattern mining: [Sample](https://udaylab.github.io/PAMI/stablePeriodicPatterns.html)

| Basic                                                                                                                                                                                                                                                           | TopK  |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------|
| SPP-growth <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/stablePeriodicPatterns/basic/SPPGrowth.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> | TSPIN <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/stablePeriodicPatterns/topk/TSPIN.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>|
| SPP-ECLAT <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/stablePeriodicPatterns/basic/SPPECLAT.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>   |       |

#### 2.7. Recurring pattern mining: [Sample](https://udaylab.github.io/PAMI/RecurringPatterns.html)

| Basic                                                                                                                                                                                                                                               |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| RPgrowth <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/recurringPatterns/basic/RPGrowth.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |


### 3. Mining patterns from binary Geo-referenced (or spatiotemporal) databases

#### 3.1. Geo-referenced frequent pattern mining: [Sample](https://udaylab.github.io/PAMI/frequentSpatialPatternMining.html)

| Basic                                                                                                                                                                                                                                                                   |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| spatialECLAT  <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/georeferencedFrequentPattern/basic/SpatialECLAT.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |
| FSP-growth <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/georeferencedFrequentPattern/basic/FSPGrowth.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>       |

#### 3.2. Geo-referenced periodic frequent pattern mining: [Sample](https://udaylab.github.io/PAMI/periodicFrequentSpatial.html)

| Basic                                                                                                                                                                                                                                                                    |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| GPFPMiner <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/georeferencedPeriodicFrequentPattern/basic/GPFPMiner.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |
| PFS-ECLAT <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/georeferencedPeriodicFrequentPattern/basic/PFS_ECLAT.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |
| ST-ECLAT <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/georeferencedPeriodicFrequentPattern/basic/STECLAT.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>    |

#### 3.3. Geo-referenced partial periodic pattern mining:[Sample](https://udaylab.github.io/PAMI/partialPeriodicSpatialPatternMining.html)

| Basic                                                                                                                                                                                                                                                                |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| STECLAT <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/georeferencedPeriodicFrequentPattern/basic/STECLAT.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |


### 4. Mining patterns from Utility (or non-binary) databases

#### 4.1. High utility pattern mining:   [Sample](https://udaylab.github.io/PAMI/highUtilityPatternMining.html)

| Basic    |
|----------|
| EFIM  <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/highUtilityPattern/basic/EFIM.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>   |
| HMiner <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/highUtilityPattern/basic/HMiner.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |
| UPGrowth <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/highUtilityPattern/basic/UPGrowth.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>|

#### 4.2. High utility frequent pattern mining:  [Sample](https://udaylab.github.io/PAMI/highUtiltiyFrequentPatternMining.html)

| Basic                                                                                                                                                                                                                                                   |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| HUFIM <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/highUtilityFrequentPatterns/basic/HUFIM.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |


#### 4.3. High utility geo-referenced frequent pattern mining:  [Sample](https://udaylab.github.io/PAMI/highUtilitySpatialPatternMining.html)

| Basic                                                                                                                                                                                                                                                                 |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| SHUFIM <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/highUtilityGeoreferencedFrequentPattern/basic/SHUFIM.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |

#### 4.4. High utility spatial pattern mining:  [Sample](https://udaylab.github.io/PAMI/highUtilitySpatialPatternMining.html)

| Basic                                                                                                                                                                                                                                                    | topk                                                                                                                                                                                                                                                     |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| HDSHIM <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/highUtilitySpatialPattern/basic/HDSHUIM.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> | TKSHUIM <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/highUtilitySpatialPattern/topK/TKSHUIM.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |
| SHUIM  <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/highUtilitySpatialPattern/basic/SHUIM.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>   |


#### 4.5. Relative High utility pattern mining: [Sample](https://github.com/UdayLab/PAMI/blob/main/sampleManuals/mainManuals/relativeUtility.html)

| Basic                                                                                                                                                                                                                                                  |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| RHUIM <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/relativeHighUtilityPattern/basic/RHUIM.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> | 


#### 4.6. Weighted frequent pattern mining: [Sample](https://udaylab.github.io/PAMI/weightedFrequentPattern.html)

| Basic                                                                                                                                                                                                                                               |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| WFIM  <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/weightedFrequentPatterns/basic/WFIM.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |


#### 4.7. Weighted frequent regular pattern mining: [Sample](https://udaylab.github.io/PAMI/weightedFrequentRegularPatterns.html)

| Basic                                                                                                                                                                                                                                                          |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| WFRIMiner <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/weightedFrequentRegularPatterns/basic/WFRI.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |


#### 4.8. Weighted frequent neighbourhood pattern mining: [Sample](https://github.com/UdayLab/PAMI/blob/main/docs/weightedSpatialFrequentPattern.html)

| Basic       |
|-------------|
| SSWFPGrowth |

### 5. Mining  patterns from fuzzy transactional/temporal/geo-referenced databases
#### 5.1. Fuzzy Frequent pattern mining: [Sample](https://github.com/UdayLab/PAMI/fuzzyFrequentPatternMining.html)

| Basic                                                                                                                                                                                                                                                   |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| FFI-Miner <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/fuzzyFrequentPattern/basic/FFIMiner.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |




#### 5.2. Fuzzy correlated pattern mining: [Sample](https://udaylab.github.io/PAMI/fuzzyCorrelatedPatternMining.html)

| Basic                                                                                                                                                                                                                                                       |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| FCP-growth <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/fuzzyCorrelatedPattern/basic/FCPGrowth.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |


#### 5.3. Fuzzy geo-referenced frequent pattern mining: [Sample](https://github.com/UdayLab/PAMI/fuzzyFrequentSpatialPatternMining.html)

| Basic                                                                                                                                                                                                                                                                  |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| FFSP-Miner <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/fuzzyGeoreferencedFrequentPattern/basic/FFSPMiner.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |

#### 5.4. Fuzzy periodic frequent pattern mining: [Sample](https://github.com/UdayLab/PAMI/fuzzyPeriodicFrequentPatternMining.html)

| Basic                                                                                                                                                                                                                                                             |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| FPFP-Miner <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/fuzzyPeriodicFrequentPattern/basic/FPFPMiner.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |

#### 5.5. Fuzzy geo-referenced periodic frequent pattern mining: [Sample](https://github.com/UdayLab/PAMI/fuzzySpatialPeriodicFrequentPatternMining.html)

| Basic                                                                                                                                                                                                                                                                                                 |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| FGPFP-Miner (under development) <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/fuzzyGeoreferencedPeriodicFrequentPattern/basic/FGPFP_miner.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |

### 6. Mining patterns from uncertain transactional/temporal/geo-referenced databases


#### 6.1. Uncertain frequent pattern mining: [Sample](https://udaylab.github.io/PAMI/uncertainFrequentPatternMining.html)

| Basic                                                                                                                                                                                                                                                   | top-k |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------|
| PUF <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/uncertainFrequentPatterns/basic/PUFGrowth.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> | TUFP  |
| TubeP  <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/uncertainFrequentPatterns/basic/TubeP.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>  |       |
| TubeS  <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/uncertainFrequentPatterns/basic/TubeS.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>  |       | 
| UVEclat                                                                                                                                                                                                                                                 |       |

#### 6.2. Uncertain periodic frequent pattern mining: [Sample](https://udaylab.github.io/PAMI/uncertainPeriodicFrequentPatternMining.html)

| Basic                                                                                                                                                                                                                                                                          |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| UPFP-growth <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/uncertainPeriodicFrequentPatterns/basic/UPFPGrowth.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>       |
| UPFP-growth++ <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/uncertainPeriodicFrequentPatterns/basic/UPFPGrowthPlus.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |


#### 6.3. Uncertain Weighted frequent pattern mining: [Sample](https://udaylab.github.io/PAMI/weightedUncertainFrequentPatterns.html)

| Basic                                                                                                                                                                                                                                                 |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| WUFIM <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/uncertainWeightedFrequent/basic/WUFIM.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |

### 7. Mining patterns from sequence databases

#### 7.1. Sequence frequent pattern mining: [Sample](https://github.com/UdayLab/PAMI/blob/main/docs/weightedSpatialFrequentPattern.html)
    
| Basic                                                                                                                                                                                                                                                       |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| SPADE <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/sequencePatternMining/basic/SPADE.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>           |
| PrefixSpan <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/sequencePatternMining/basic/prefixSpan.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |


#### 7.2. Geo-referenced Frequent Sequence Pattern mining

| Basic                                                                                                                                                                                                                                                                                         |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| GFSP-Miner (under development)<a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/georeferencedFrequentSequencePattern/basic/GFSP_miner.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |

### 8. Mining patterns from multiple timeseries databases

#### 8.1. Partial periodic pattern mining (under development)

| Basic                                                                                                                                                                                                                                                                                             |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| PP-Growth (under development) <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/partialPeriodicPatternInMultipleTimeSeries/basic/PPGrowth.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |

## 9. Mining interesting patterns from Streams

1. Frequent pattern mining

| Basic         |
|---------------|
 | to be written |

2. High utility pattern mining

| Basic |
|-------|
 | HUPMS |


## 10. Mining patterns from contiguous character sequences (E.g., DNA, Genome, and Game sequences)

#### 10.1. Contiguous Frequent Patterns

| Basic                                                                                                                                                                                                                                                                   |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| PositionMining <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/contiguousFrequentPattern/basic/PositionMining.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |

## 11. Mining pattrens from Graphs
__coming soon__
  
  
     

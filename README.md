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
[![pages-build-deployment](https://github.com/UdayLab/PAMI/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/UdayLab/PAMI/actions/workflows/pages/pages-build-deployment)
[![Dependabot Updates](https://github.com/UdayLab/PAMI/actions/workflows/dependabot/dependabot-updates/badge.svg)](https://github.com/UdayLab/PAMI/actions/workflows/dependabot/dependabot-updates)
[![CodeQL](https://github.com/UdayLab/PAMI/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/UdayLab/PAMI/actions/workflows/github-code-scanning/codeql)

[Click here for more information](https://pepy.tech/project/pami)


***

# Table of Contents

- [Introduction](#introduction)
- [Development process](#process-flow-chart)
- [Inputs and outputs of a PAMI algorithm](#inputs-and-outputs-of-an-algorithm-in-pami)
- [Recent updates](#recent-updates)
- [Features](#features)
- [Maintenance](#Maintenance)
- [Try your first PAMI program](#try-your-first-PAMI-program)
- [Evaluation](#evaluation)
- [Reading Material](#Reading-Material)
- [License](#License)
- [Documentation](#Documentation)
- [Background](#Background)
- [Getting Help](#Getting-Help)
- [Discussion and Development](#Discussion-and-Development)
- [Contribution to PAMI](#Contribution-to-PAMI)
- [Tutorials](#tutorials)
  - [Association rule mining](#0-association-rule-mining)
  - [Mining transactional databases](#1-pattern-mining-in-binary-transactional-databases)
  - [Mining temporal databases](#2-pattern-mining-in-binary-temporal-databases)
  - [Mining spatiotemporal databases](#3-mining-patterns-from-binary-geo-referenced-or-spatiotemporal-databases)
  - [Mining utility databases](#4-mining-patterns-from-utility-or-non-binary-databases)
  - [Mining fuzzy databases](#5-mining--patterns-from-fuzzy-transactionaltemporalgeo-referenced-databases)
  - [Mining uncertain databases](#6-mining-patterns-from-uncertain-transactionaltemporalgeo-referenced-databases)
  - [Mining sequence databases](#7-mining-patterns-from-sequence-databases)
  - [Mining multiple timeseries](#8-mining-patterns-from-multiple-timeseries-databases)
  - [Mining streams](#9-mining-interesting-patterns-from-streams)
  - [Mining character sequences](#10-mining-patterns-from-contiguous-character-sequences-eg-dna-genome-and-game-sequences)
  - [Mining graphs](#11-mining-patterns-from-graphs)
  - [Additional features](#12-additional-features)
    - [Synthetic data generator](#121-creation-of-synthetic-databases)
    - [Dataframes to databases](#122-converting-a-dataframe-into-a-specific-database-type)
    - [Gathering database statistics](#123-gathering-the-statistical-details-of-a-database)
- [Real-World Case Studies](#real-world-case-studies)


***
# Introduction

PAttern MIning (PAMI) is a Python library containing several algorithms to discover user interest-based patterns in a wide-spectrum of datasets across multiple computing platforms. Useful links to utilize the services of this library were provided below:
NAME:SANGEETH

1. Youtube tutorial https://www.youtube.com/playlist?list=PLKP768gjVJmDer6MajaLbwtfC9ULVuaCZ

2. Tutorials (Notebooks) https://github.com/UdayLab/PAMI/tree/main/notebooks
   
3. User manual https://udaylab.github.io/PAMI/manuals/index.html

4. Coders manual https://udaylab.github.io/PAMI/codersManual/index.html

5. Code documentation https://pami-1.readthedocs.io

6. Datasets   https://u-aizu.ac.jp/~udayrage/datasets.html

7. Discussions on PAMI usage https://github.com/UdayLab/PAMI/discussions

8. Report issues https://github.com/UdayLab/PAMI/issues

***
# Flow Chart of Developing Algorithms in PAMI

![PAMI's production process](./images/pamiDevelopmentSteps.png?raw=true)

<!--- ![alt text](https://github.com/[username]/[reponame]/blob/[branch]/image.jpg?raw=true) ---> 
***
# Inputs and Outputs of an Algorithm in PAMI

![Inputs and Outputs](./images/inputOutputPAMIalgo.png?raw=true)
***
# Recent Updates

- **Version 2024.07.02:** 
In this latest version, the following updates have been made:
  - Included one new algorithms, **PrefixSpan**, for Sequential Pattern.
  - Optimized the following pattern mining algorithms: **PFPGrowth, PFECLAT, GPFgrowth and PPF_DFS**.
  - Test cases are implemented for the following algorithms, **Contiguous Frequent patterns, Correlated Frequent Patterns, Coverage Frequent Patterns, Fuzzy Correlated Frequent Patterns, Fuzzy Frequent Patterns, Fuzzy Georeferenced Patterns, Georeferenced Frequent Patterns, Periodic Frequent Patterns, Partial Periodic Frequent Patterns, HighUtility Frequent Patterns, HighUtility Patterns, HighUtility Georeferenced Frequent Patterns, Frequent Patterns, Multiple Minimum Frequent Patterns, Periodic Frequent Patterns, Recurring Patterns, Sequential Patterns, Uncertain Frequent Patterns, Weighted Uncertain Frequent Patterns**.
  - The algorithms mentioned below are automatically tested, **Frequent Patterns, Correlated Frequent Patterns, Contiguous Frequent patterns, Coverage Frequent Patterns, Recurring Patterns, Sequential Patterns**.

Total number of algorithms: 89

***
# Features

- ✅ Tested to the best of our possibility
- 🔋 Highly optimized to our best effort, light-weight, and energy-efficient
- 👀 Proper code documentation
- 🍼 Ample examples of using various algorithms at [./notebooks](https://github.com/UdayLab/PAMI/tree/main/notebooks) folder
- 🤖 Works with AI libraries such as TensorFlow, PyTorch, and sklearn. 
- ⚡️ Supports Cuda and PySpark 
- 🖥️ Operating System Independence
- 🔬 Knowledge discovery in static data and streams
- 🐎 Snappy
- 🐻 Ease of use

***

# Maintenance

  __Installation__
  
  1. Installing basic pami package (recommended)

         pip install pami

  2. Installing pami package in a GPU machine that supports CUDA

         pip install 'pami[gpu]'

  3. Installing pami package in a distributed network environment supporting Spark

         pip install 'pami[spark]'

  4. Installing pami package for developing purpose

         pip install 'pami[dev]'

  5. Installing complete Library of pami

         pip install 'pami[all]'

  __Upgradation__

  
        pip install --upgrade pami
  

  __Uninstallation__

  
        pip uninstall pami 
       

  __Information__ 


        pip show pami

***
# *Try your first PAMI program*

```shell
$ python
```

```python
# first import pami 
from PAMI.frequentPattern.basic import FPGrowth as alg
fileURL = "https://u-aizu.ac.jp/~udayrage/datasets/transactionalDatabases/Transactional_T10I4D100K.csv"
minSup=300
obj = alg.FPGrowth(iFile=fileURL, minSup=minSup, sep='\t')
#obj.startMine()  #deprecated
obj.mine()
obj.save('frequentPatternsAtMinSupCount300.txt')
frequentPatternsDF= obj.getPatternsAsDataFrame()
print('Total No of patterns: ' + str(len(frequentPatternsDF))) #print the total number of patterns
print('Runtime: ' + str(obj.getRuntime())) #measure the runtime
print('Memory (RSS): ' + str(obj.getMemoryRSS()))
print('Memory (USS): ' + str(obj.getMemoryUSS()))
```

```
Output:
Frequent patterns were generated successfully using frequentPatternGrowth algorithm
Total No of patterns: 4540
Runtime: 8.749667644500732
Memory (RSS): 522911744
Memory (USS): 475353088
```

***

# Evaluation:

1. we compared three different Python libraries such as PAMI, mlxtend and efficient-apriori for Apriori.
2. (Transactional_T10I4D100K.csv)is a transactional database downloaded from PAMI and
used as an input file for all libraries.
3. Minimum support values and seperator are also same.

* The performance of the **Apriori algorithm** is shown in the graphical results below:
1. Comparing the **Patterns Generated** by different Python libraries for the Apriori algorithm:

   <img width="573" alt="Screenshot 2024-04-11 at 13 31 31" src="https://github.com/vanithakattumuri/PAMI/assets/134862983/fd7974bc-ffe2-44dd-82e3-a5306a8a23bd">
   
2. Evaluating the **Runtime** of the Apriori algorithm across different Python libraries:

   <img width="567" alt="Screenshot 2024-04-11 at 13 31 20" src="https://github.com/vanithakattumuri/PAMI/assets/134862983/5d615ae3-dc0d-49ba-a880-4890bb1f11c5">

3. Comparing the **Memory Consumption** of the Apriori algorithm across different Python libraries:

   <img width="570" alt="Screenshot 2024-04-11 at 13 31 08" src="https://github.com/vanithakattumuri/PAMI/assets/134862983/5d5991ca-51ae-442d-9b5e-2d21bbebfedd">

For more information, we have uploaded the evaluation file in two formats:
- One **ipynb** file format, please check it here. [Evaluation File ipynb](https://github.com/UdayLab/PAMI/blob/main/notebooks/Evaluation-neverDelete.ipynb) 
- Two **pdf** file format, check here. [Evaluation File Pdf](https://github.com/UdayLab/PAMI/blob/main/notebooks/evaluation.pdf)

***
# Reading Material

For more examples, refer this YouTube link [YouTube](https://www.youtube.com/playlist?list=PLKP768gjVJmDer6MajaLbwtfC9ULVuaCZ)

***
# License

[![GitHub license](https://img.shields.io/github/license/UdayLab/PAMI)](https://github.com/UdayLab/PAMI/blob/main/LICENSE)
***

# Documentation

The official documentation is hosted on [PAMI](https://pami-1.readthedocs.io).
***

# Background

The idea and motivation to develop PAMI was from [Kitsuregawa Lab](https://www.tkl.iis.u-tokyo.ac.jp/new/resources?lang=en) at the University of Tokyo. Work on ``PAMI`` started at [University of Aizu](https://u-aizu.ac.jp/en/) in 2020 and
has been under active development since then.

***
# Getting Help

For any queries, the best place to go to is Github Issues [GithubIssues](https://github.com/orgs/UdayLab/discussions/categories/q-a).

***
# Discussion and Development

In our GitHub repository, the primary platform for discussing development-related matters is the university lab. We encourage our team members and contributors to utilize this platform for a wide range of discussions, including bug reports, feature requests, design decisions, and implementation details.

***
# Contribution to PAMI

We invite and encourage all community members to contribute, report bugs, fix bugs, enhance documentation, propose improvements, and share their creative ideas.

***
# Tutorials
### 0. Association Rule Mining

| Basic                                                                                                                                                                                                                                                |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Confidence <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/associationRules/basic/confidence.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| Lift <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/associationRules/basic/lift.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>             |
| Leverage <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/associationRules/basic/leverage.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>     |

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

## 11. Mining patterns from Graphs

#### 11.1. Frequent sub-graph mining
| Basic                                                                                                                                                                                                                                      | topk                                                                                                                                                                                                                                  |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
 | Gspan <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/subgraphMining/basic/gspan.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> | TKG <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/subgraphMining/topk/tkg.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |

#### 11.2. Graph transactional coverage pattern mining
| Basic                                                                                                                                                                                                                                                       |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| GTCP<a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/graphTransactionalCoveragePatterns/basic/GTCP.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |

## 12. Additional Features

#### 12.1. Creation of synthetic databases

| Database type                                                                                                                                                                                                                                                                        |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Transactional database <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/extras/syntheticDataGenerators/TransactionalDatabase.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | |
| Temporal database <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/extras/syntheticDataGenerators/TemporalDatabase.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>           |
| Utility database (coming soon)                                                                                                                                                                                                                                                       |
| spatio-transactional database (coming soon)                                                                                                                                                                                                                                          |
| spatio-temporal database (coming soon)                                                                                                                                                                                                                                               |
| fuzzy transactional database (coming soon)                                                                                                                                                                                                                                           |
| fuzzy temporal database (coming soon)                                                                                                                                                                                                                                                |
| Sequence database generator (coming soon)                                                                                                                                                                                                                                            |


#### 12.2. Converting a dataframe into a specific database type
| Approaches                                                                                                                                                                                                                                                   |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Dense dataframe to databases <a target="_blank" href="https://colab.research.google.com/github/udayLab/PAMI/blob/main/notebooks/extras/DF2DB/denseDF2DB.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| Sparse dataframe to databases (coming soon)                                                                                                                                                                                                                  |


#### 12.3. Gathering the statistical details of a database
| Approaches                                                                                                                                                                                                                                                        |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Transactional database <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/extras/stats/TransactionalDatabase.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| Temporal database <a target="_blank" href="https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/extras/stats/TemporalDatabase.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>           |
| Utility database (coming soon)                                                                                                                                                                                                                                    |

#### 12.4. Convertors
| Approaches                 |
|----------------------------|
| Subgraphs2FlatTransactions |
| CSV2Parquet                |
| CSV2BitInteger             |
| CSV2Integer                |



#### 12.4. Generating Latex code for the experimental results
| Approaches               |
|--------------------------|
| Latex code (coming soon) |

***

# Real World Case Studies

1. Air pollution analytics <a target="_blank" href="https://colab.research.google.com/github/udayLab/PAMI/blob/main/notebooks/airPollutionAnalytics.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


[Go to Top](#table-of-contents)

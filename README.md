![PyPI](https://img.shields.io/pypi/v/PAMI)
![AppVeyor](https://img.shields.io/appveyor/build/udayRage/PAMI)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/PAMI)
![GitHub all releases](https://img.shields.io/github/downloads/udayRage/PAMI/total)
[![GitHub license](https://img.shields.io/github/license/udayRage/PAMI)](https://github.com/udayRage/PAMI/blob/main/LICENSE)
![PyPI - Implementation](https://img.shields.io/pypi/implementation/PAMI)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/PAMI)
![PyPI - Status](https://img.shields.io/pypi/status/PAMI)
[![GitHub issues](https://img.shields.io/github/issues/udayRage/PAMI)](https://github.com/udayRage/PAMI/issues)
[![GitHub forks](https://img.shields.io/github/forks/udayRage/PAMI)](https://github.com/udayRage/PAMI/network)
[![GitHub stars](https://img.shields.io/github/stars/udayRage/PAMI)](https://github.com/udayRage/PAMI/stargazers)



# Introduction
PAMI stands for PAttern MIning. It constitutes several pattern mining algorithms to discover interesting patterns in transactional/temporal/spatiotemporal databases.
This software is provided under [GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007](https://www.gnu.org/licenses/quick-guide-gplv3.html).

1. The user manual for PAMI library is available at https://udayrage.github.io/PAMI/index.html
2. Datasets to implement PAMI algorithms are available at https://www.u-aizu.ac.jp/~udayrage/software.html
3. Please report issues in the software at https://github.com/udayRage/PAMI/issues
  
  
# Installation

       pip install pami
       
# Details 
Total available algorithms: 43

1. Frequent pattern mining: 
     
   | Basic | Closed | Maximal | Top-k |
   |-------|--------|---------|-------|
   |Apriori|Closed|maxFP-growth|topK|
   |FP-growth|    |   | |
   |ECLAT| | | |
   |ECLAT-bitSet| | | |

2. Frequent pattern mining using other measures:
    
    |Basic|
    |-----|
    |RSFP|
        
3. Correlated pattern mining: 

    |Basic|
    |-----|
    |CP-growth|
    |CP-growth++|
    
4. Frequent spatial pattern mining: 

    |Basic|
    |-----|
    |spatialECLAT|
    |FSP-growth ?|
    
5. Correlated spatial pattern mining: 

    |Basic|
    |-----|
    |SCP-growth|
    
6. Fuzzy correlated pattern mining:

    |Basic|
    |-----|
    |CFFI|
    
7. Fuzzy frequent spatial pattern mining:

    |Basic|
    |-----|
    |FFSI|
    
8. Fuzzy periodic frequent pattern mining:

    |Basic|
    |-----|
    |FPFP-Miner|
    
9. High utility frequent spatial pattern mining:

    |Basic|
    |-----|
    |HDSHUIM|
 
10. High utility pattern mining:

     |Basic|
     |-----|
     |EFIM|
     |UPGrowth|
    
11. Partial periodic frequent pattern:

    |Basic|
    |-----|
    |GPF-growth|
    |PPF-DFS|
    
12. Periodic frequent pattern mining: 

    |Basic| Closed | Maximal |
    |-----|--------|---------|
    |PFP-growth|CPFP|maxPF-growth|
    |PFP-growth++| | |
    |PS-growth| | |
    |PFP-ECLAT| | |
    
13. Partial periodic pattern mining:

    |Basic|Maximal|
    |-----|-------|
    |3P-growth|max3P-growth|
    |3PECLAT| |
    

14. Uncertain correlated pattern mining: 
    
    |Basic|
    |-----|
    |CFFI|
    
15. Uncertain frequent pattern mining:
    
    |Basic|
    |-----|
    |PUF|
    |TubeP|
    |TubeS|
    
16. Uncertain periodic frequent pattern mining:
     
     |Basic|
     |-----|
     |PTubeP|
     |PTubeS|
     |UPFP-growth|
     
17. Local periodic pattern mining:

     |Basic|
     |-----|
     |LPPMbredth|
     |LPPMdepth|
     |LPPGrowth|
18. Recurring pattern mining:

     |Basic|
     |-----|
     |RPgrowth|
 
     



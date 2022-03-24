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
  
  
  __Contact us by Discord__ https://discord.gg/9WgKkrSJ
  
# Installation

       pip install pami
       
# Upgrade
      
       pip install --upgrade pami
       
# Details 
Total available algorithms: 43

1. Frequent pattern mining: 
     
   | Basic | Closed | Maximal | Top-k | CUDA | pyspark |
   |-------|--------|---------|-------|------|--------|
   |Apriori|Closed|maxFP-growth|topK|cudaAprioriGCT|parallelApriori|
   |FP-growth|    |   | |cudaAprioriTID|parallelFPGrowth|
   |ECLAT| | | |cudaEclatGCT|parallelECLAT|
   |ECLAT-bitSet| | | | | |
   |ECLAT-diffset|  | | | |

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
    |FSP-growth|
    
5. Correlated spatial pattern mining: 

    |Basic|
    |-----|
    |CSP-growth|
    
6. Fuzzy correlated pattern mining:

    |Basic|
    |-----|
    |FCP-growth|

7. Fuzzy Frequent pattern mining:

    |Basic|
    |-----|
    |FFI-Miner|
    
8. Fuzzy frequent spatial pattern mining:

    |Basic|
    |-----|
    |FFSP-Miner|
    
9. Fuzzy periodic frequent pattern mining:

    |Basic|
    |-----|
    |FPFP-Miner|

10. High utility frequent pattern mining:

    |Basic|
    |-----|
    |HUFIM|
    
11. High utility frequent spatial pattern mining:

    |Basic|
    |-----|
    |SHUFIM|
 
12. High utility pattern mining:

     |Basic|
     |-----|
     |EFIM|
     |HMiner|
     |UPGrowth|
     
13. High utility spatial pattern mining:

     |Basic|topk|
     |-----|----|
     |HDSHIM|TKSHUIM|
     |SHUIM|
     
14. Local periodic pattern mining:

     |Basic|
     |-----|
     |LPPGrowth|
     |LPPMBreadth|
     |LPPMDepth|
    
15. Partial periodic frequent pattern:

    |Basic|
    |-----|
    |GPF-growth|
    |PPF-DFS|
    
16. Periodic frequent pattern mining: 

    |Basic| Closed | Maximal |
    |-----|--------|---------|
    |PFP-growth|CPFP|maxPF-growth|
    |PFP-growth++| | |
    |PS-growth| | |
    |PFP-ECLAT| | |
    
17. Partial periodic pattern mining:

    |Basic|Closed|Maximal|topk|
    |-----|-------|------|----|
    |3P-growth|3P-close|max3P-growth|Topk_3Pgrowth|
    |3PECLAT| | | | |
    

18. Periodic correlated pattern mining: 
    
    |Basic|
    |-----|
    |EPCP-growth|
    

19. Uncertain correlated pattern mining: 
    
    |Basic|
    |-----|
    |CFFI|
    
20. Uncertain frequent pattern mining:
    
    |Basic| top-k |
    |-----|-----|
    |PUF|TUFP|
    |TubeP| |
    |TubeS| | 
    |UVEclat| |
    
21. Uncertain periodic frequent pattern mining:
     
     |Basic|
     |-----|
     |PTubeP|
     |PTubeS|
     |UPFP-growth|
     
22. Recurring pattern mining:

     |Basic|
     |-----|
     |RPgrowth|
     
23. Relative High utility pattern mining: 
    
    |Basic|
    |-----|
    |RHUIM|
 
24. Stable periodic pattern mining: 
    
    |Basic|
    |-----|
    |SPP-growth|
    
25. Uncertain correlated pattern mining: 
    
    |Basic|
    |-----|
    |CFFI|
 
     
 
     
     



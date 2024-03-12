Periodic frequent pattern mining involves identifying patterns that occur at regular intervals within a temporal database, where each record represents an event or observation associated with a specific timestamp.
In this context, a pattern is considered periodic-frequent if it satisfies user-defined constraints on both the minimum support (minSup) and maximum periodicity (maxPer).
The goal is to discover patterns that exhibit regular recurring behavior over time, providing insights into temporal trends, cyclic phenomena, or periodic events within the dataset.
Unlike traditional frequent pattern mining, which focuses on static datasets, periodic frequent pattern mining specifically targets temporal databases,
where time-related attributes play a crucial role in pattern discovery and analysis.

Applications: Temporal Data Analysis, Healthcare Monitoring, Retail Sales Forecasting, Network Traffic Analysis.


Basic
======


.. toctree::
   :maxdepth: 1

   periodicFrequentPatternbasicPFPGrowth
   periodicFrequentPatternbasicPFPGrowthPlus
   periodicFrequentPatternbasicPSGrowth
   periodicFrequentPatternbasicPFECLAT
   periodicFrequentPatternbasicPFPMC

closed
=======


.. toctree::
   :maxdepth: 1

   periodicFrequentPatternclosedCPFPMiner

maximal
========


.. toctree::
   :maxdepth: 1

   periodicFrequentPatternmaximalMaxPFGrowth


Top-K
=======


.. toctree::
   :maxdepth: 1

   periodicFrequentPatterntopkkPFPMinerkPFPMiner
   periodicFrequentPatterntopkTopkPFPTopkPFP




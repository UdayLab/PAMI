# User Manual for Implementing the Algorithms in PAMI Library

 
Chapter 1: Introduction

   1. [About PAMI: Motivation and the supporting people](./aboutPAMI.html)
   2. [Maintenance of PAMI library](./installation.html)
   3. [Organization of algorithms in PAMI](./organization.html)

Chapter 2: Preparation of various datasets (or databases)
   1. [Transactional database](./transactionalDatabase.html)
   2. [Temporal database](./temporalDatabase.html)
   3. [Utility database](./utilityDatabase.html)
   4. [Uncertain database](uncertainDatabases.md)
   5. [Location/spatial database](locationDatabase.md)
   6. [Neighborhood database](./neighborhoodDatabase.html)
   7. [Geo-referenced database](./spatialDatabase.html)
   8. [Multiple timeseries data](./timeSeries.html)

Chapter 3: Converting dataframes to databases

   1. [Format of dense dataframe](./denseDataFrame.html) 
   2. [Format of sparse dataframe](./sparseDataFrame.html)
   3. [Approaches to convert a dataframe into various database formats](./DenseFormatDF.html)
   4. [An advanced approach to convert a dataframe into a database](./DF2DBPlus.html)

Chapter 4: Creation of very large synthetic databases
  
   1. [Creation of transactional database](./createTransactionalDatabase.html)
   2. [Creation of temporal database](./createTemporalDatabase.html)
   3. [Creation of Geo-referenced database (under development)](./createSpatiotemporalDatabase.html) 

Chapter 5: Printing, displaying, and saving the statistical details of a database
 
   1. [Transactional databases](./TransactionalDatabase.html)      
   2. [Temporal database (partially under development)](./temporalDatabaseStats.html)
   3. [Utility database (partially under development)](./utilityDatabaseStats.html)   
   4. [Uncertain database (under development)](./uncertainDatabaseStats.html)
   5. [Geo-referenced database (under development)](./geoReferencedDatabase.html)

Chapter 6: Implementing algorithms in PAMI

   1. [Directly executing PAMI algorithms on a terminal/command prompt](./terminalExecute.html)
   2. [Mining interesting patterns using an algorithm](./utilization.html)
   3. [Evaluating multiple pattern mining algorithms](./evaluateMultipleAlgorithms.html)

   __Note:__ Click on the 'Basic' and 'Adv' links of an algorithm in the [Github page](https://github.com/udayRage/PAMI) to know more about its usage.

Chapter 7: Visualization in PAMI

   1. [Visualization of spatial items](visualizeSpatialItems.html)

Chapter 8: Additional topics

   1. [Generating latex graphs for publishing results in conferences and journals (under development)](./generateLatexGraphs.html)
   2. [Creation of neighborhood file from a Geo-referenced database using Euclidean distance](./neighborFileFromspatialDataframe.html)
   
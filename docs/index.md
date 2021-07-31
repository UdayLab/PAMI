**[CLICK HERE](manual.html)** to access the PAMI manual.

# User Manual 
Key concepts in each link were briefly mentioned to save your valuable time. Click on the necessary links to know more.

1. [About PAMI: Motivation and the people who supported](aboutPAMI.html)
   
   PAMI is a PAttern MIning Python library to discover hidden patterns in Big Data.

1. [Installation/Update/uninstall PAMI](installation.html)
   
         pip install pami
   
1. [Organization of Algorithms in PAMI](organization.html)
   
   The algorithms in PAMI are organized in a hierarchical structure as follows: 
   
        PAMI.theoriticalModel.basic/maximal/closed/topk.algorithmName
   
1. [Creating Databases](createDatabases.html)
   
    1. [Transactional database](transactionalDatabase.html)
       
            format: item1<sep>item2<sep>...<sep>itemN
       
    1. [Temporal database](temporalDatabase.html)

            format: timestamp<sep>item1<sep>item2<sep>...<sep>itemN
    1. [Neighborhood database](neighborhoodDatabase.html)
            
            format: spatialItem1<sep>spatialItem3<sep>spatialItem10<sep>...
       
    1. [Utility database](utilityDatabase.html)
       
            format: item1<sep>...<sep>itemN:totalUtility:utilityItem1<sep>...<sep>utilityItemN
    
    Default separator used in PAMI is tab space. However, users can override the separator with their choice.
   
1. [Converting Dataframes to Databases](df2db.html)
   1. [Dense dataframe to database](denseDF2DB.html)
   1. [Sparse dataframe to database](sparseDF2DB.html)
   1. [Spatiotemporal dataframe to databases](stDF2DB.html)
   
1. [Exceuting Algorithms in PAMI](utilization.html)    
   1. Importing PAMI algorithms into your program
   1. Executing PAMI algorithms directly on the terminal
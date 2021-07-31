**[CLICK HERE](manual.html)** to access the PAMI manual.


# Organization of Algorithms in PAMI

Several theoretical models (e.g., frequent pattern model, correlated pattern model, and periodic pattern model) have been explored in the literature to find 
user interest-based patterns in a database.  For each theoretical model, there exists several algorithms to find desired patterns in the data effectively. 

Depending on the type of patterns that are being discovered from the data, the mining algorithms of a theoretical model can be broadly classified into the following three types:

1. **Basic pattern mining algorithms:** These are the fundamental algorithms that aim to discover all patterns that satisfy the 
   user-specified constraints in a database. These algorithms are generally computationally expensive and produce too many patterns.
   
1. **Closed pattern mining algorithms:** The basic algorithms often produce too many patterns most of which may be uninteresting
   to the user depending on the user and/or application applications. Moreover, the basic algorithms are computationally expensive
   due to the generation of too many patterns. To confront this problem, the concept of closed patterns was explored in the literature 
   to find only a minimal subset of patterns, called closed patterns, from which all desired patterns can be later generated.

1. **Maximal pattern mining algorithms:**  Although closed pattern mining algorithms produce relatively fewer patterns than the basic 
   pattern mining algorithms, they still produce too many patterns most of which may be uninteresting to the user. In this context, the 
   notion of maximal patterns wax explored to find only the longest patterns whose subsets by default are also interesting patterns in the database.
   From the perspective of applied research, finding maximal patterns in the database has the following advances:
   1. Generation of fewer patterns: In many applications, users are interested in finding only the longest patterns that
   satisfy the specified constraints.  Maximal pattern mining algorithms provide exactly this information to the users by supressing
      their subsets information.
   1. Computationally less expensive: Finding fewer patterns in a database reduces the memory and runtime requirements of the mining process.
   
   __**Note:**__ Unlike closed pattern mining algorithms, maximal pattern mining algorithms are lossy by nature. That is, we loose the
   information regarding all subsets of a maximal pattern. Thus, we advise users discretion in choosing the right algorithm.
   
1. **Top-k pattern mining algorithms:** The basic, closed, and maximal pattern mining algorithms require the user-specified constraints for the execution.
   In many scenarios, specifying these constraints is an open research problem. Researchers 
   have introduced top-k pattern mining algorithms to address this open research problem. Given the k value by the user, the objective of these algorithms is
   to find only those top-k patterns that have highest values in a database. 
   
# Structuring of Algorithms in PAMI

In the PAMI library, the mining algorithms have been packaged in a hierarchical fashion. The first hierarchical level is 'PAMI'.
The second hierarchical level is the name of the theoretical model. The third hierarchical level is the type of the mining algorithm.
The last hierarhical level contains the mining algorithms.
   
1. PAMI
   * theoreticalModel1
      * basic
         * Algo1
         * Algo2
         ...
         * AlgoM
      * closed
         * cAlgo1
         * cAlgo2 ...
         * cAlgoM
      * maximal
         * mAlgo1
         * mAlgo2 ...
         * mAlgoM
      * top-k
        * kAlgo1
        * kAlgo2 ...
        * kAlgoM
   * theoreticalModel2
      * basic
         * Algo1
         * Algo2
         ...
         * AlgoM      
       
   * ...
   * theoreticalModelN
      * basic
         * Algo1
         * Algo2
         ...
         * AlgoM
      * closed
         * cAlgo1
         * cAlgo2 ...
         * cAlgoM
      * maximal
         * mAlgo1
         * mAlgo2 ...
         * mAlgoM
      * top-k
        * kAlgo1
        * kAlgo2 ...
        * kAlgoM

**NOTE:**  The  closed, maximal, and top-k packages exist for a theoretical model if and only if there exist the corresponding mining algorithm in the PAMI library. 

The user can import a pattern mining algorithm using the following syntax:

    PAMI.theoriticalModel.basic/closed/maximal/topk import Algorithm as algo
   
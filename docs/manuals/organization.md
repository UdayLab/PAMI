[Previous](installation.html)|[🏠 Home](index.html)|[Next](transactionalDatabase.html)

## Structural Organization  of Algorithms in PAMI

In PAMI, the mining algorithms have been packaged in a hierarchical fashion. The first hierarchical level is 'PAMI'.
The second hierarchical level is the name of the theoretical model of an interesting pattern. The third hierarchical level is the type of the mining algorithm.
The last hierarchical level contains the mining algorithms. The organizational structure of algorithms in PAMI is shown below:

      PAMI
        |-theoriticalModel (e.g., frequent/periodicFrequent/highUtility patterns)
                 |-patternType (e.g., basic/maximal/closed/topk)
                          |-algorithmName
                          

## Importing an Algorithm in PAMI

The user can import a pattern mining algorithm using the following syntax:

    PAMI.theoriticalModel.patternType import Algorithm as algo


## Pseudo-structural Organization of Algorithms in PAMI
A hypothetical structural arrangement of algorithms in PAMI is shown below.

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
    * extras
      * graphs
        * ...
      * ...

**NOTE:**  The  closed, maximal, and top-k packages exist for a theoretical model if and only if there exist the corresponding mining algorithm in the PAMI library. 

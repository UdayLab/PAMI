[__<--__ Return to home page](index.html)

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
    * ...
    * extras
      * graphs
        * ...
      * ...

**NOTE:**  The  closed, maximal, and top-k packages exist for a theoretical model if and only if there exist the corresponding mining algorithm in the PAMI library. 


## Tips to Writing an Algorithm

The algorithm developers were requested to check the following points before writing a new code:
1. If you are developing an algorithm for a theoretical pattern mining model, please check whether the corresponding model exists in PAMI or not. 
2. If the folder for the theoretical model already exists in PAMI, then check for the next level and so on. 
3. If the folder for the theoretical model does not exist in PAMI, then create a new folder with the name. Please create sub-folders for lower levels in PAMI hierarchy.
4. If the abstract class file exists, then please use the abstract file in your code. Otherwise, create an abstract file and use it in your algorithm. 
[__<--__ Return to home page](index.html)

## Location to place an algorithm in PAMI

### Hierarchical structure of algorithms
The algorithm in PAMI follow an hierarchical structure as shown below.

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
      
### Steps to place an algorithm for a new pattern model
If the user is writing the code for new theoretical pattern model that does not exist in PAMI, then please perform the following steps:

   1. Create a directory with name of the pattern and enter into the corresponding directory.
   2. Generate an empty file, titled __\_\_init\_\_.py__
   3. Create another directory, say basic, closed, or maximal, depending on the type of pattern.
   4. Enter into this subdirectory and generate another empty file, titled __\_\_init\_\_.py__
   5. Place your __abstract.py__ file in this subdirectory.
   6. Place your algorithm inheriting the __abstract.py__ file also in this subdirectory. If your algorithm does not inherit the abstract.py file, no problem. Simply place your algorithm here. 

### Steps to place an algorithm for the already existing pattern model
If the user has written an algorithm for a pattern model that already exists in the PAMI repository, then perform the following steps:

   1. Enter into the directory of the corresponding theoretical pattern model. 
   2. Enter into the subdirectory, say basic, of the corresponding model. If the subdirectory for your algorithm does not exist, then create a subdirectory, and create a \_\_init\_\_.py file.
   3. Place your algorithm in the subdirectory.
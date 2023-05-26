PAMI.stablePeriodicFrequentPattern.basic package
================================================

Submodules
----------

PAMI.stablePeriodicFrequentPattern.basic.SPPECLAT module
--------------------------------------------------------

.. automodule:: PAMI.stablePeriodicFrequentPattern.basic.SPPECLAT
   :members:
   :undoc-members:
   :show-inheritance:


**Description:**

    Stable periodic pattern mining aims to dicover all interesting patterns in a temporal database using three contraints minimum support,
    maximum period and maximum lability, that have support no less than the user-specified minimum support  constraint and lability no
    greater than maximum lability.

**Reference:**

        Fournier-Viger, P., Yang, P., Lin, J. C.-W., Kiran, U. (2019). Discovering Stable Periodic-Frequent Patterns in Transactional Data. Proc.
         32nd Intern. Conf. on Industrial, Engineering and Other Applications of Applied Intelligent Systems (IEA AIE 2019), Springer LNAI, pp. 230-244

**Attributes:**

        iFile : file
            Name of the Input file or path of the input file
        oFile : file
            Name of the output file or path of the output file
        minSup: int or float or str
            The user can specify minSup either in count or proportion of database size.
            If the program detects the data type of minSup is integer, then it treats minSup is expressed in count.
            Otherwise, it will be treated as float.
            Example: minSup=10 will be treated as integer, while minSup=10.0 will be treated as float
        maxPer: int or float or str
            The user can specify maxPer either in count or proportion of database size.
            If the program detects the data type of maxPer is integer, then it treats maxPer is expressed in count.
            Otherwise, it will be treated as float.
            Example: maxPer=10 will be treated as integer, while maxPer=10.0 will be treated as float
        maxLa: int or float or str
            The user can specify maxLa either in count or proportion of database size.
            If the program detects the data type of maxLa is integer, then it treats maxLa is expressed in count.
            Otherwise, it will be treated as float.
            Example: maxLa=10 will be treated as integer, while maxLa=10.0 will be treated as float
        sep : str
            This variable is used to distinguish items from one another in a transaction. The default seperator is tab space or \t.
            However, the users can override their default separator.
        memoryUSS : float
            To store the total amount of USS memory consumed by the program
        memoryRSS : float
            To store the total amount of RSS memory consumed by the program
        startTime:float
            To record the start time of the mining process
        endTime:float
            To record the completion time of the mining process
        Database : list
            To store the transactions of a database in list
        mapSupport : Dictionary
            To maintain the information of item and their frequency
        lno : int
            it represents the total no of transactions
        tree : class
            it represents the Tree class
        itemSetCount : int
            it represents the total no of patterns
        finalPatterns : dict
            it represents to store the patterns
        tidList : dict
            stores the timestamps of an item

**Methods:**

        startMine()
            Mining process will start from here
        getPatterns()
            Complete set of patterns will be retrieved with this function
        save(oFile)
            Complete set of periodic-frequent patterns will be loaded in to a output file
        getPatternsAsDataFrame()
            Complete set of periodic-frequent patterns will be loaded in to a dataframe
        getMemoryUSS()
            Total amount of USS memory consumed by the mining process will be retrieved from this function
        getMemoryRSS()
            Total amount of RSS memory consumed by the mining process will be retrieved from this function
        getRuntime()
            Total amount of runtime taken by the mining process will be retrieved from this function
        creatingItemSets()
            Scan the database and store the items with their timestamps which are periodic frequent
        calculateLa()
            Calculates the support and period for a list of timestamps.
        Generation()
            Used to implement prefix class equivalence method to generate the periodic patterns recursively



**Methods to execute code on terminal**

        Format:
                  >>>   python3 SPPECLAT.py <inputFile> <outputFile> <minSup> <maxPer> <maxLa>

        Example:
                  >>>    python3 SPPECLAT.py sampleDB.txt patterns.txt 10.0 4.0 2.0

        .. note:: constraints will be considered in percentage of database transactions

**Importing this algorithm into a python program**

.. code-block:: python

                from PAMI.stablePeriodicFrequentPattern.basic import SPPECLAT as alg

                obj = alg.PFPECLAT("../basic/sampleTDB.txt", 5, 3, 3)

                obj.startMine()

                Patterns = obj.getPatterns()

                print("Total number of Stable Periodic Frequent Patterns:", len(Patterns))

                obj.save("patterns")

                Df = obj.getPatternsAsDataFrame()

                memUSS = obj.getMemoryUSS()

                print("Total Memory in USS:", memUSS)

                memRSS = obj.getMemoryRSS()

                print("Total Memory in RSS", memRSS)

                run = obj.getRuntime()

                print("Total ExecutionTime in seconds:", run)

**Credits:**

         The complete program was written by  P.Likhitha under the supervision of Professor Rage Uday Kiran.


PAMI.stablePeriodicFrequentPattern.basic.SPPGrowth module
---------------------------------------------------------

.. automodule:: PAMI.stablePeriodicFrequentPattern.basic.SPPGrowth
   :members:
   :undoc-members:
   :show-inheritance:


**Methods to execute code on terminal**

        Format:
                  >>>   python3 SPPGrowth.py <inputFile> <outputFile> <minSup> <maxPer> <maxLa>
        Example:
                  >>>  python3 SPPGrowth.py sampleTDB.txt patterns.txt 0.3 0.4 0.3

        .. note:: constraints will be considered in percentage of database transactions

**Importing this algorithm into a python program**

.. code-block:: python

        from PAMI.stablePeriodicFrequentPattern.basic import SPPGrowth as alg

                obj = alg.SPPGrowth(iFile, minSup, maxPer, maxLa)

                obj.startMine()

                Patterns = obj.getPatterns()

                print("Total number of Stable Periodic Frequent Patterns:", len(Patterns))

                obj.save(oFile)

                Df = obj.getPatternsAsDataFrame()

                memUSS = obj.getMemoryUSS()

                print("Total Memory in USS:", memUSS)

                memRSS = obj.getMemoryRSS()

                print("Total Memory in RSS", memRSS)

                run = obj.getRuntime()

                print("Total ExecutionTime in seconds:", run)

**Credits:**

         The complete program was written by  P.Likhitha under the supervision of Professor Rage Uday Kiran.


PAMI.stablePeriodicFrequentPattern.basic.SPPGrowthDump module
-------------------------------------------------------------

.. automodule:: PAMI.stablePeriodicFrequentPattern.basic.SPPGrowthDump
   :members:
   :undoc-members:
   :show-inheritance:


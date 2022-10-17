PAMI.partialPeriodicFrequentPattern.basic package
=================================================

Submodules
----------

PAMI.partialPeriodicFrequentPattern.basic.GPFgrowth module
----------------------------------------------------------

.. automodule:: PAMI.partialPeriodicFrequentPattern.basic.GPFgrowth
   :members:
   :undoc-members:
   :show-inheritance:

**Methods to execute code on terminal**

        Format:
                  >>>   python3 GPFgrowth.py <inputFile> <outputFile> <minSup> <maxPer> <minPR>
        Example:
                  >>>   python3 GPFgrowth.py sampleDB.txt patterns.txt 10 10 0.5

        .. note:: minSup will be considered in percentage of database transactions


**Importing this algorithm into a python program**

.. code-block:: python

        from PAMI.partialPeriodicFrequentPattern.basic import GPFgrowth as alg

        obj = alg.GPFgrowth(inputFile, outputFile, minSup, maxPer, minPR)

        obj.startMine()

        partialPeriodicFrequentPatterns = obj.getPatterns()

        print("Total number of partial periodic Patterns:", len(partialPeriodicFrequentPatterns))

        obj.savePatterns(oFile)

        Df = obj.getPatternInDf()

        memUSS = obj.getMemoryUSS()

        print("Total Memory in USS:", memUSS)

        memRSS = obj.getMemoryRSS()

        print("Total Memory in RSS", memRSS)

        run = obj.getRuntime()

        print("Total ExecutionTime in seconds:", run)

**Credits:**

         The complete program was written by So Nakamura  under the supervision of Professor Rage Uday Kiran.




PAMI.partialPeriodicFrequentPattern.basic.PPF\_DFS module
---------------------------------------------------------

.. automodule:: PAMI.partialPeriodicFrequentPattern.basic.PPF_DFS
   :members:
   :undoc-members:
   :show-inheritance:


**Methods to execute code on terminal**

        Format:
                  >>>   python3 PPF_DFS.py <inputFile> <outputFile> <minSup> <maxPer> <minPR>
        Example:
                  >>>   python3 PPF_DFS.py sampleDB.txt patterns.txt 10 10 0.5

        .. note:: minSup will be considered in percentage of database transactions


**Importing this algorithm into a python program**

.. code-block:: python

        from PAMI.partialPeriodicFrequentpattern.basic import PPF_DFS as alg

        obj = alg.PPF_DFS(iFile, minSup)

        obj.startMine()

        frequentPatterns = obj.getPatterns()

        print("Total number of Frequent Patterns:", len(frequentPatterns))

        obj.savePatterns(oFile)

        Df = obj.getPatternInDataFrame()

        memUSS = obj.getMemoryUSS()

        print("Total Memory in USS:", memUSS)

        memRSS = obj.getMemoryRSS()

        print("Total Memory in RSS", memRSS)

        run = obj.getRuntime()

        print("Total ExecutionTime in seconds:", run)

**Credits:**

         The complete program was written by So Nakamura  under the supervision of Professor Rage Uday Kiran.



PAMI.partialPeriodicFrequentPattern.basic.abstract module
---------------------------------------------------------

.. automodule:: PAMI.partialPeriodicFrequentPattern.basic.abstract
   :members:
   :undoc-members:
   :show-inheritance:

Module contents
---------------

.. automodule:: PAMI.partialPeriodicFrequentPattern.basic
   :members:
   :undoc-members:
   :show-inheritance:

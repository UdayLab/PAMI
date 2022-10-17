PAMI.periodicFrequentPattern.maximal package
============================================

Submodules
----------

PAMI.periodicFrequentPattern.maximal.MaxPFGrowth module
-------------------------------------------------------

.. automodule:: PAMI.periodicFrequentPattern.maximal.MaxPFGrowth
   :members:
   :undoc-members:
   :show-inheritance:
**Methods to execute code on terminal**

        Format:
                  >>>  python3 maxpfrowth.py <inputFile> <outputFile> <minSup> <maxPer>
        Example:
                  >>>  python3 maxpfrowth.py sampleTDB.txt patterns.txt 0.3 0.4

        .. note:: minSup will be considered in percentage of database transactions


**Importing this algorithm into a python program**

.. code-block:: python

            from PAMI.periodicFrequentPattern.maximal import MaxPFGrowth as alg

            obj = alg.MaxPFGrowth("../basic/sampleTDB.txt", "2", "6")

            obj.startMine()

            Patterns = obj.getPatterns()

            print("Total number of Frequent Patterns:", len(periodicFrequentPatterns))

            obj.savePatterns("patterns")

            Df = obj.getPatternsAsDataFrame()

            memUSS = obj.getMemoryUSS()

            print("Total Memory in USS:", memUSS)

            memRSS = obj.getMemoryRSS()

            print("Total Memory in RSS", memRSS)

            run = obj.getRuntime()

            print("Total ExecutionTime in seconds:", run)

**Credits:**

         The complete program was written by  P.Likhitha under the supervision of Professor Rage Uday Kiran.

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

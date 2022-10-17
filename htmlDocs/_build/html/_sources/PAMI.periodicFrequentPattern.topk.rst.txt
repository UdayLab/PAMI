PAMI.periodicFrequentPattern.topk package
=========================================

Submodules
----------

PAMI.periodicFrequentPattern.topk.TOPKPeriodic module
-----------------------------------------------------

.. automodule:: PAMI.periodicFrequentPattern.topk.TOPKPeriodic
   :members:
   :undoc-members:
   :show-inheritance:


**Methods to execute code on terminal**

        Format:
                  >>>  python3 TOPKPeriodic.py <inputFile> <outputFile> <k>
        Example:
                  >>>  python3 TOPKPeriodic.py sampleTDB.txt patterns.txt 0.3

        .. note:: minSup will be considered in percentage of database transactions


**Importing this algorithm into a python program**

.. code-block:: python

            from PAMI.periodicFrequentPattern.topk import TOPKPeriodic as alg

            obj = alg.TOPKPeriodic("../basic/sampleTDB.txt", "10")

            obj.startMine()

            periodicFrequentPatterns = obj.getPatterns()

            print("Total number of Periodic Frequent Patterns:", len(periodicFrequentPatterns))

            obj.savePatterns(oFile)

            Df = obj.getPatternsAsDataFrame()

            memUSS = obj.getMemoryUSS()

            print("Total Memory in USS:", memUSS)

            memRSS = obj.getMemoryRSS()

            print("Total Memory in RSS", memRSS)

            run = obj.getRuntime()

            print("Total ExecutionTime in seconds:", run)
**Credits:**

         The complete program was written by  P.Likhitha  under the supervision of Professor Rage Uday Kiran.

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


PAMI.periodicFrequentPattern.topk.TopkPFP module
------------------------------------------------

.. automodule:: PAMI.periodicFrequentPattern.topk.TopkPFP
   :members:
   :undoc-members:
   :show-inheritance:


**Methods to execute code on terminal**

        Format:
                  >>>  python3 TopkPFP.py <inputFile> <outputFile> <k> <maxPer>
        Example:
                  >>>  python3 TopkPFP.py sampleDB.txt patterns.txt 10 3

        .. note:: minSup will be considered in percentage of database transactions


**Importing this algorithm into a python program**

.. code-block:: python

            import PAMI.periodicFrequentPattern.topk.TopkPFPGrowth as alg

            obj = alg.TopkPFPGrowth(iFile, k, maxPer)

            obj.startMine()

            periodicFrequentPatterns = obj.getPatterns()

            print("Total number of Periodic Frequent Patterns:", len(periodicFrequentPatterns))

            obj.savePatterns(oFile)

            Df = obj.getPatternsAsDataFrame()

            memUSS = obj.getMemoryUSS()

            print("Total Memory in USS:", memUSS)

            memRSS = obj.getMemoryRSS()

            print("Total Memory in RSS", memRSS)

            run = obj.getRuntime()

            print("Total ExecutionTime in seconds:", run)
**Credits:**

         The complete program was written by  P.Likhitha  under the supervision of Professor Rage Uday Kiran.

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



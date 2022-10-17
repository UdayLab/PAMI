PAMI.recurringPattern package
=============================

Submodules
----------

PAMI.recurringPattern.RPGrowth module
-------------------------------------

.. automodule:: PAMI.recurringPattern.RPGrowth
   :members:
   :undoc-members:
   :show-inheritance:

**Methods to execute code on terminal**

        Format:
                  >>>  python3 RPGrowth.py <inputFile> <outputFile> <maxPer> <minPS> <minRec>
        Example:
                  >>>  python3 RPGrowth.py sampleTDB.txt patterns.txt 0.3 0.4 2

        .. note:: maxPer and minPS will be considered in percentage of database transactions


**Importing this algorithm into a python program**

.. code-block:: python

            from PAMI.periodicFrequentPattern.recurring import RPGrowth as alg

            obj = alg.RPGrowth(iFile, maxPer, minPS, minRec)

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

         The complete program was written by   C. Saideep  under the supervision of Professor Rage Uday Kiran.

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


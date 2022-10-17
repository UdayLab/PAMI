PAMI.partialPeriodicPattern.timeSeries package
==============================================

Submodules
----------

PAMI.partialPeriodicPattern.timeSeries.PPGrowth module
------------------------------------------------------

.. automodule:: PAMI.partialPeriodicPattern.timeSeries.PPGrowth
   :members:
   :undoc-members:
   :show-inheritance:


**Methods to execute code on terminal**

        Format:
                  >>>  python3 PPGrowth.py <inputFile> <outputFile> <minSup> <maxPer>
        Example:
                  >>>  python3 PPGrowth.py sampleTDB.txt patterns.txt 0.3 0.4

        .. note:: minSup will be considered in percentage of database transactions


**Importing this algorithm into a python program**

.. code-block:: python

            from PAMI.periodicFrequentPattern.basic import PPGrowth as alg

            obj = alg.PPGrowth(iFile, minSup, maxPer)

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



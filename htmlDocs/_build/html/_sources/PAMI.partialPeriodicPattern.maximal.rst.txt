PAMI.partialPeriodicPattern.maximal package
===========================================

Submodules
----------

PAMI.partialPeriodicPattern.maximal.Max3PGrowth module
------------------------------------------------------

.. automodule:: PAMI.partialPeriodicPattern.maximal.Max3PGrowth
   :members:
   :undoc-members:
   :show-inheritance:


**Methods to execute code on terminal**

        Format:
                  >>>  python3 max3prowth.py <inputFile> <outputFile> <periodicSupport> <period>
        Example:
                  >>>  python3 Max3PGrowth.py sampleTDB.txt patterns.txt 3 4

        .. note:: periodicSupport will be considered in count


**Importing this algorithm into a python program**

.. code-block:: python

            from PAMI.periodicFrequentPattern.maximal import ThreePGrowth as alg

            obj = alg.ThreePGrowth(iFile, periodicSupport, period)

            obj.startMine()

            partialPeriodicPatterns = obj.partialPeriodicPatterns()

            print("Total number of partial periodic Patterns:", len(partialPeriodicPatterns))

            obj.savePatterns(oFile)

            Df = obj.getPatternInDf()

            memUSS = obj.getMemoryUSS()

            print("Total Memory in USS:", memUSS)

            memRSS = obj.getMemoryRSS()

            print("Total Memory in RSS", memRSS)

            run = obj.getRuntime()

            print("Total ExecutionTime in seconds:", run)

**Credits:**

         The complete program was written by  P.Likhitha  under the supervision of Professor Rage Uday Kiran.

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



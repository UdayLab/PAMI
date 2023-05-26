PAMI.stablePeriodicFrequentPattern.topK package
===============================================

Submodules
----------

PAMI.stablePeriodicFrequentPattern.topK.TSPIN module
----------------------------------------------------

.. automodule:: PAMI.stablePeriodicFrequentPattern.topK.TSPIN
   :members:
   :undoc-members:
   :show-inheritance:



**Methods to execute code on terminal**

        Format:
                  >>>   python3 TSPIN.py <inputFile> <outputFile> <maxPer> <maxLa>
        Example:
                  >>>  python3 TSPIN.py sampleTDB.txt patterns.txt 0.3 0.4 0.6

        .. note:: maxPer, maxLa and k will be considered in percentage of database transactions



**Importing this algorithm into a python program**

.. code-block:: python

            from PAMI.stablePeriodicFrequentPattern.basic import TSPIN as alg

            obj = alg.TSPIN(iFile, maxPer, maxLa, k)

            obj.startMine()

            stablePeriodicFrequentPatterns = obj.getPatterns()

            print("Total number of Periodic Frequent Patterns:", len(stablePeriodicFrequentPatterns))

            obj.savePatterns(oFile)

            Df = obj.getPatternsAsDataFrame()

            memUSS = obj.getMemoryUSS()

            print("Total Memory in USS:", memUSS)

            memRSS = obj.getMemoryRSS()

            print("Total Memory in RSS", memRSS)

            run = obj.getRuntime()

            print("Total ExecutionTime in seconds:", run)

**Credits:**

         The complete program was written by P.Likhitha  under the supervision of Professor Rage Uday Kiran.


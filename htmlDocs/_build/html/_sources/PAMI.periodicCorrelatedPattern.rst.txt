PAMI.periodicCorrelatedPattern package
======================================

Submodules
----------

PAMI.periodicCorrelatedPattern.EPCPGrowth module
------------------------------------------------

.. automodule:: PAMI.periodicCorrelatedPattern.EPCPGrowth
   :members:
   :undoc-members:
   :show-inheritance:



**Methods to execute code on terminal**

        Format:
                  >>>   python3 PFPGrowth.py <inputFile> <outputFile> <minSup> <maxPer> <sep>
        Example:
                  >>>  python3 PFPGrowth.py sampleTDB.txt patterns.txt 0.3 0.4

        .. note:: minSup and maxPer will be considered in percentage of database transactions


**Importing this algorithm into a python program**

.. code-block:: python

            from PAMI.periodicCorrelatedPattern.basic import EPCPGrowth as alg

            obj = alg.EPCPGrowth(iFile, minSup, minAllCOnf, maxPer, maxPerAllConf)

            obj.startMine()

            periodicCorrelatedPatterns = obj.getPatterns()

            print("Total number of Periodic Frequent Patterns:", len(periodicCorrelatedPatterns))

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

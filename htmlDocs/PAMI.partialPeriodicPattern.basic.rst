PAMI.partialPeriodicPattern.basic package
=========================================

Submodules
----------


PAMI.partialPeriodicPattern.basic.PPPGrowth module
--------------------------------------------------

.. automodule:: PAMI.partialPeriodicPattern.basic.PPPGrowth
   :members:
   :undoc-members:
   :show-inheritance:


**Methods to execute code on terminal**

        Format:
                  >>>  python3 PPPGrowth.py <inputFile> <outputFile> <periodicSupport> <period>
        Example:
                  >>>  python3 PPPGrowth.py sampleDB.txt patterns.txt 10.0 2.0

        .. note:: periodicSupprot and period will be considered in count


**Importing this algorithm into a python program**

.. code-block:: python

        from PAMI.fuzzyPeriodicFrequentPattern.basic import FPFPMiner as alg

        obj =alg.FPFPMiner("input.txt",2,3)

        obj.startMine()

        periodicFrequentPatterns = obj.getPatterns()

        print("Total number of Fuzzy Periodic Frequent Patterns:", len(periodicFrequentPatterns))

        obj.savePatterns("output.txt")

        memUSS = obj.getMemoryUSS()

        print("Total Memory in USS:", memUSS)

        memRSS = obj.getMemoryRSS()

        print("Total Memory in RSS", memRSS)

        run = obj.getRuntime()

        print("Total ExecutionTime in seconds:", run)

**Credits:**

         The complete program was written by   P.Likhitha  under the supervision of Professor Rage Uday Kiran.




PAMI.partialPeriodicPattern.basic.PPP_ECLAT module
---------------------------------------------------

.. automodule:: PAMI.partialPeriodicPattern.basic.PPP_ECLAT
   :members:
   :undoc-members:
   :show-inheritance:


**Methods to execute code on terminal**

        Format:
                  >>>  python3 PPP_ECLAT.py <inputFile> <outputFile> <periodicSupport> <period>
        Example:
                  >>>  python3 PPP_ECLAT.py sampleDB.txt patterns.txt 0.3 0.4

        .. note:: periodicSupport and period will be considered in percentage of database transactions

**Importing this algorithm into a python program**

.. code-block:: python

        from PAMI.periodicFrequentPattern.basic import PPP_ECLAT as alg

        obj = alg.PPP_ECLAT(iFile, periodicSupport,period)

        obj.startMine()

        Patterns = obj.getPatterns()

        print("Total number of partial periodic patterns:", len(Patterns))

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




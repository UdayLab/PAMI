PAMI.weightedUncertainFrequentPattern.basic package
===================================================

Submodules
----------

PAMI.weightedUncertainFrequentPattern.basic.WUFIM module
--------------------------------------------------------

.. automodule:: PAMI.weightedUncertainFrequentPattern.basic.WUFIM
   :members:
   :undoc-members:
   :show-inheritance:

**Methods to execute code on terminal**

        Format:
                  >>>  python3 WUFIM.py <inputFile> <outputFile> <minSup>
        Example:
                  >>>  python3 WUFIM.py sampleTDB.txt patterns.txt 3

                 .. note:: minSup  will be considered in support count or frequency

**Importing this algorithm into a python program**

.. code-block:: python

        from PAMI.weightedUncertainFrequentPattern.basic import WFIM as alg

        obj = alg.WFIM(iFile, wFile, expSup, expWSup)

        obj.startMine()

        Patterns = obj.getPatterns()

        print("Total number of  Patterns:", len(Patterns))

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
         


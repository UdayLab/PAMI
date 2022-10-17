PAMI.frequentPattern.closed package
===================================

Submodules
----------

PAMI.frequentPattern.closed.CHARM module
----------------------------------------

.. automodule:: PAMI.frequentPattern.closed.CHARM
   :members:
   :undoc-members:
   :show-inheritance:


**Methods to execute code on terminal**

        Format:
                  >>> python3 CHARM.py <inputFile> <outputFile> <minSup>

        Example:
                  >>> python3 CHARM.py sampleDB.txt patterns.txt 10.0

        .. note:: minSup will be considered in percentage of database transactions


**Importing this algorithm into a python program**

.. code-block:: python

        from PAMI.frequentPattern.closed import closed as alg

        obj = alg.Closed(iFile, minSup)

        obj.startMine()

        frequentPatterns = obj.getPatterns()

        print("Total number of Closed Frequent Patterns:", len(frequentPatterns))

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

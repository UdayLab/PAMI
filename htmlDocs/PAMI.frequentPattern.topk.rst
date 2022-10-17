PAMI.frequentPattern.topk package
=================================

Submodules
----------

PAMI.frequentPattern.topk.FAE module
------------------------------------

.. automodule:: PAMI.frequentPattern.topk.FAE
   :members:
   :undoc-members:
   :show-inheritance:



**Methods to execute code on terminal**

        Format:
                  >>> python3 FAE.py <inputFile> <outputFile> <minSup>

        Example:
                  >>> python3 FAE.py sampleDB.txt patterns.txt 10

        .. note:: minSup will be considered in percentage of database transactions


**Importing this algorithm into a python program**

.. code-block:: python

            import PAMI.frequentPattern.topK.FAE as alg

            obj = alg.FAE(iFile, minSup)

            obj.startMine()

            frequentPatterns = obj.getPatterns()

            print("Total number of Frequent Patterns:", len(frequentPatterns))

            obj.savePatterns(oFile)

            Df = obj.getPatternInDataFrame()

            memUSS = obj.getMemoryUSS()

            print("Total Memory in USS:", memUSS)

            memRSS = obj.getMemoryRSS()

            print("Total Memory in RSS", memRSS)

            run = obj.getRuntime()

            print("Total ExecutionTime in seconds:", run)


**Credits:**

         The complete program was written by P.Likhitha  under the supervision of Professor Rage Uday Kiran.



PAMI.relativeHighUtilityPatterns.basic package
==============================================

Submodules
----------

PAMI.relativeHighUtilityPatterns.basic.RHUIM module
---------------------------------------------------

.. automodule:: PAMI.relativeHighUtilityPatterns.basic.RHUIM
   :members:
   :undoc-members:
   :show-inheritance:


**Methods to execute code on terminal**

        Format:
                  >>> python3 RHUIM.py <inputFile> <outputFile> <minUtil> <sep>
        Example:
                  >>>  python3 RHUIM.py sampleTDB.txt output.txt 35 20


**Importing this algorithm into a python program**

.. code-block:: python

        from PAMI.relativeHighUtilityPatterns.basic import RHUIM as alg

        obj=alg.RHUIM("input.txt", 35, 20)

        obj.startMine()

        frequentPatterns = obj.getPatterns()

        print("Total number of Frequent Patterns:", len(frequentPatterns))

        obj.savePatterns(oFile)

        Df = obj.getPatternsAsDataFrame()

        memUSS = obj.getmemoryUSS()

        print("Total Memory in USS:", memUSS)

        memRSS = obj.getMemoryRSS()

        print("Total Memory in RSS", memRSS)

        run = obj.getRuntime()

        print("Total ExecutionTime in seconds:", run)
**Credits:**

         The complete program was written by  Pradeep Pallikila  under the supervision of Professor Rage Uday Kiran.


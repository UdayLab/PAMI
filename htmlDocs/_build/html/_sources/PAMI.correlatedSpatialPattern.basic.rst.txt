PAMI.correlatedSpatialPattern.basic package
===========================================

Submodules
----------

PAMI.correlatedSpatialPattern.basic.CSPGrowth module
----------------------------------------------------

.. automodule:: PAMI.correlatedSpatialPattern.basic.CSPGrowth
   :members:
   :undoc-members:
   :show-inheritance:


**Methods to execute code on terminal**

        Format:
                  >>> python3 CSPGrowth.py <inputFile> <outputFile> <neighbourFile> <minSup> <minAllConf> <sep>
        Example:
                  >>>  python3 CSPGrowth.py sampleTDB.txt output.txt sampleN.txt 0.25 0.2

                 .. note:: minSup will be considered in percentage of database transactions

**Importing this algorithm into a python program**

.. code-block:: python

        from PAMI.correlatedSpatialPattern.basic import CSPGrowth as alg

        obj = alg.CSPGrowth(iFile, frequentPatternsFile, measure, threshold)

        obj.startMine()

        Rules = obj.getPatterns()

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

         The complete program was written by B.Sai Chitra under the supervision of Professor Rage Uday Kiran.

PAMI.AssociationRules package
=============================

Submodules
----------

PAMI.AssociationRules.RuleMiner module
--------------------------------------

.. automodule:: PAMI.AssociationRules.RuleMiner
   :members:
   :undoc-members:
   :show-inheritance:


**Methods to execute code on terminal**

        Format:
                  >>>  python3 RuleMiner.py <inputFile> <outputFile> <measure> <threshold>
        Example:
                  >>>  python3 RuleMiner.py sampleTDB.txt rules.txt 'lift' 0.5

                 .. note:: measure can be lift or leverage or confidence

**Importing this algorithm into a python program**

.. code-block:: python

        from PAMI.AssociationRules import RuleMiner as alg

        obj = alg.RuleMiner(iFile, frequentPatternsFile, measure, threshold)

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

         The complete program was written by P.Likhitha  under the supervision of Professor Rage Uday Kiran.

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


PAMI.weightedFrequentRegularPattern.basic package
=================================================

Submodules
----------

PAMI.weightedFrequentRegularPattern.basic.WFRIMiner module
----------------------------------------------------------

.. automodule:: PAMI.weightedFrequentRegularPattern.basic.WFRIMiner
   :members:
   :undoc-members:
   :show-inheritance:



**Methods to execute code on terminal**

        Format:
                  >>> python3 WFRIMiner.py <inputFile> <outputFile> <weightSupport> <regularity>
        Example:
                  >>>  python3 WFRIMiner.py sampleDB.txt patterns.txt 10 5

                 .. note:: WS & regularity will be considered in support count or frequency

**Importing this algorithm into a python program**

.. code-block:: python

        from PAMI.weightedFrequentRegularpattern.basic import WFRIMiner as alg

        obj = alg.WFRIMiner(iFile, WS, regularity)

        obj.startMine()

        weightedFrequentRegularPatterns = obj.getPatterns()

        print("Total number of Frequent Patterns:", len(weightedFrequentRegularPatterns))

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

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


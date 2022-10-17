PAMI.sequentialPatternMining package
====================================

Submodules
----------


PAMI.sequentialPatternMining.prefixSpan module
----------------------------------------------

.. automodule:: PAMI.sequentialPatternMining.prefixSpan
   :members:
   :undoc-members:
   :show-inheritance:


**Methods to execute code on terminal**

        Format:
                  >>>
        Example:
                  >>>

        .. note::

**Importing this algorithm into a python program**

.. code-block:: python



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

         The complete program was written by        under the supervision of Professor Rage Uday Kiran.


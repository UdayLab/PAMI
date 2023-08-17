import matplotlib.pyplot as plt
import pandas as _pd

class plotGraphsFromDataFrame():

    def __init__(self, dataFrame):

        self._dataFrame = dataFrame

    def plotGraphsFromDataFrame(self):
        self._dataFrame.plot(x='minSup', y='patterns', kind='line')
        plt.show()
        print('Graph for No Of Patterns is successfully generated!')
        self._dataFrame.plot(x='minSup', y='runtime', kind='line')
        plt.show()
        print('Graph for Runtime taken is successfully generated!')
        self._dataFrame.plot(x='minSup', y='memory', kind='line')
        plt.show()
        print('Graph for memory consumption is successfully generated!')




if __name__ == '__main__':
    #data = {'algorithm': ['FPGrowth','FPGrowth', 'FPGrowth', 'FPGrowth', 'FPGrowth', 'ECLAT', 'ECLAT', 'ECLAT', 'ECLAT', 'ECLAT'],
    #        'minSup': [0.01, 0.02, 0.03, 0.04, 0.05, 0.01, 0.02, 0.03, 0.04, 0.05],
    #        'patterns': [386, 155, 60, 36, 10, 386, 155, 60, 26, 10],
    #        'runtime': [7.351629, 4.658654 , 4.658654 , 1.946843, 1.909376, 4.574833, 2.514252, 1.834948, 1.889892, 1.809999],
    #        'memory': [426545152, 309182464, 241397760, 225533952, 220950528, 233537536, 267165696, 252841984, 245690368,
    #                    295710720]
    #        }
    '''data = {
        'algorithm': ['FPGrowth', 'FPGrowth', 'FPGrowth', 'FPGrowth', 'FPGrowth'],
        'minSup': [0.01, 0.02, 0.03, 0.04, 0.05],
        'patterns': [386, 155, 60, 36, 10],
        'runtime': [7.351629, 4.658654, 4.658654, 1.946843, 1.909376],
        'memory': [426545152, 309182464, 241397760, 225533952, 220950528]
        }'''
    dataFrame = _pd.DataFrame(data)
    ab = plotGraphsFromDataFrame(dataFrame)
    ab.plotGraphsFromDataFrame()

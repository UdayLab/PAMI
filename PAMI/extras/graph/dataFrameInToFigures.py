import plotly.express as _px
import pandas as _pd

class dataFrameInToFigures():

    def __init__(self, dataFrame):

        self._dataFrame = dataFrame

    def plotGraphsFromDataFrame(self,xColumn='minSup',yColumn='patterns',lineLabels='algorithm'):
        fig = _px.line(self._dataFrame, x=xColumn, y=yColumn, color=lineLabels)
        fig.show()






if __name__ == '__main__':
    #data = {'algorithm': ['FPGrowth','FPGrowth', 'FPGrowth', 'FPGrowth', 'FPGrowth', 'ECLAT', 'ECLAT', 'ECLAT', 'ECLAT', 'ECLAT'],
    #        'minSup': [0.01, 0.02, 0.03, 0.04, 0.05, 0.01, 0.02, 0.03, 0.04, 0.05],
    #        'patterns': [386, 155, 60, 36, 10, 386, 155, 60, 26, 10],
    #        'runtime': [7.351629, 4.658654 , 4.658654 , 1.946843, 1.909376, 4.574833, 2.514252, 1.834948, 1.889892, 1.809999],
    #        'memory': [426545152, 309182464, 241397760, 225533952, 220950528, 233537536, 267165696, 252841984, 245690368,
    #                    295710720]
    #        }
    data = {
        'algorithm': ['FPGrowth', 'FPGrowth', 'FPGrowth', 'FPGrowth', 'FPGrowth'],
        'minSup': [0.01, 0.02, 0.03, 0.04, 0.05],
        'patterns': [386, 155, 60, 36, 10],
        'runtime': [7.351629, 4.658654, 4.658654, 1.946843, 1.909376],
        'memory': [426545152, 309182464, 241397760, 225533952, 220950528]
        }
    dataFrame = _pd.DataFrame(data)
    ab = dataFrameInToFigures(dataFrame)
    ab.plotGraphsFromDataFrame()
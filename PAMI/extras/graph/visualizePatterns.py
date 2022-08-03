import plotly.express as px
import pandas as pd

class visualizePatterns():
    
    def __init__(self, file, topk):
        self.file = file
        self.topk = topk

    def visualize(self):
        """
        Visualize points produced by pattern miner.
    
        :param file: String for file name
        :param top: visualize topk patterns
        """
    
        long = []
        lat = []
        name = []
        color = []
        R = G = B = 0
    
        lines = {}
        with open(self.file, "r") as f:
            for line in f:
                lines[line] = len(line)
            
        lines = list(dict(sorted(lines.items(), key=lambda x:x[1])[-self.topk:]).keys())

        start = 1
        for line in lines:
            start += 1
            if start % 3 == 0:
                R += 20
            if start % 3 == 1:
                G += 20
            if start % 3 == 2:
                B += 20

            if R > 255:
                R = 0
            if G > 255:
                G = 0
            if B > 255:
                B = 0

            RHex = hex(R)[2:]
            GHex = hex(G)[2:]
            BHex = hex(B)[2:]

            line = line.split(":")
            freq = line[-1]
            freq = "Frequency: " + freq.strip()
            line = line[:-1]
            points = line[0].split(" ")
            points = [x for x in points if x != ""]
            points = [x.strip("POINT(") for x in points]
            points = [x.strip(")") for x in points]

            for i in range(len(points)):
                if i % 2 == 0:
                    lat.append(float(points[i]))
                    name.append(freq)
                    color.append("#" + RHex + GHex + BHex)
                else:
                    long.append(float(points[i]))

        df = pd.DataFrame({"lon": long, "lat": lat, "freq": name, "col": color})
    
        fig = px.scatter_mapbox(df, lat="lon", lon="lat", hover_name="freq", color="col", zoom=3, height=300)
        fig.update_layout(mapbox_style="open-street-map")
        fig.show()

if __name__ == '__main__':
    obj = visualizePatterns('sensor_output.txt', 10)
    obj.visualize()

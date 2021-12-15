class predictedClass2Transaction:
    def getBinaryTransaction(self,predicted_classes,minThreshold=0.8):
        self.predictions_dict ={}
        for name, p, box in predicted_classes:
            if p > minThreshold :
                    if name not in self.predictions_dict:
                        self.predictions_dict[name] = [p, ]
                    else:
                        self.predictions_dict[name].append(p)
        return self.predictions_dict

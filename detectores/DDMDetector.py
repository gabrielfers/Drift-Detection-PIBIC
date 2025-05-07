from river.drift.binary import DDM
from detectores.DetectorDriftBase import DetectorDriftBase

class DDMDetector(DetectorDriftBase):
    def __init__(self, threshold=0.05):
        super().__init__()
        self.detector = DDM()
        self.name = "_DDM"
        self.threshold = threshold

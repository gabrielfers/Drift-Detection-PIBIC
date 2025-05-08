from river.drift import KSWIN
from detectores.DetectorDriftBase import DetectorDriftBase

class KSWINDetector(DetectorDriftBase):
    def __init__(self):
        super().__init__()
        self.detector = KSWIN()
        self.name = "_KSWIN"

from river.drift import KSWIN
from detectores.DetectorDriftBase import DetectorDriftBase

class KSWINDetector(DetectorDriftBase):
    def __init__(self):
        super().__init__()
        self.detector = KSWIN()
        self.name = "_KSWIN"

    def atualizar(self, erro):
        self.detector.update(erro)

    @property
    def drift_detectado(self):
        return self.detector.drift_detected
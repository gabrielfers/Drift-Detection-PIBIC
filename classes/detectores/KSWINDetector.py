from classes.superclasse.DetectorDriftBase import DetectorDriftBase
from river.drift import KSWIN

class KSWINDetector(DetectorDriftBase):
    def __init__(self):
        super().__init__()
        self.detector = KSWIN()

    def atualizar(self, erro):
        self.detector.update(erro)

    @property
    def drift_detectado(self):
        return self.detector.drift_detected
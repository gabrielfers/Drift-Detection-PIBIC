from river.drift.binary import DDM
from detectores.DetectorDriftBase import DetectorDriftBase

class DDMDetector(DetectorDriftBase):
    def __init__(self, threshold=0.05):
        super().__init__()
        self.detector = DDM()
        self.threshold = threshold

    def atualizar(self, erro):
        erro_binario = 1 if erro > self.threshold else 0
        self.detector.update(erro_binario)

    @property
    def drift_detectado(self):
        return self.detector.drift_detected
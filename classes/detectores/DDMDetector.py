from classes.superclasse.DetectorDriftBase import DetectorDriftBase
from river.drift.binary import DDM

class DDMDetector(DetectorDriftBase):
    def __init__(self):
        super().__init__()
        self.detector = DDM()

    def atualizar(self, erro):
        self.detector.update(erro)

    @property
    def drift_detectado(self):
        return self.detector.drift_detected
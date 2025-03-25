from classes.superclasse.DetectorDriftBase import DetectorDriftBase
from river.drift.binary import FHDDM

class FHDDMDetector(DetectorDriftBase):
    def __init__(self):
        super().__init__()
        self.detector = FHDDM()

    def atualizar(self, erro):
        self.detector.update(erro)

    @property
    def drift_detectado(self):
        return self.detector.drift_detected
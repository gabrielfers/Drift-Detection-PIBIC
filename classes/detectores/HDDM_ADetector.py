from classes.superclasse.DetectorDriftBase import DetectorDriftBase
from river.drift.binary import HDDM_A


class HDDM_ADetector(DetectorDriftBase):
    def __init__(self):
        super().__init__()
        self.detector = HDDM_A()

    def atualizar(self, erro):
        self.detector.update(erro)

    @property
    def drift_detectado(self):
        return self.detector.drift_detected
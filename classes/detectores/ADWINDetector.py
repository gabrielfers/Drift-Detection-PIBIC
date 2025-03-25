from classes.superclasse.DetectorDriftBase import DetectorDriftBase
from river.drift import ADWIN

class ADWINDetector(DetectorDriftBase):
    def __init__(self):
        super().__init__()
        self.detector = ADWIN()

    def atualizar(self, erro):
        self.detector.update(erro)

    @property
    def drift_detectado(self):
        return self.detector.drift_detected
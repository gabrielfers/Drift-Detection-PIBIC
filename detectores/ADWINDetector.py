from river.drift import ADWIN
from detectores.DetectorDriftBase import DetectorDriftBase

class ADWINDetector(DetectorDriftBase):
    def __init__(self):
        super().__init__()
        self.detector = ADWIN()

    def atualizar(self, erro):
        self.detector.update(erro)

    @property
    def drift_detectado(self):
        return self.detector.drift_detected
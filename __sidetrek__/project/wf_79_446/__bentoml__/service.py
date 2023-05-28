import numpy as np
import bentoml
from bentoml.io import NumpyNdarray

model_runner = bentoml.sklearn.get("fake_note_3:latest").to_runner()
svc = bentoml.Service("fake_note_service_3", runners=[model_runner])

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_series: np.ndarray) -> np.ndarray:
    result = model_runner.predict.run(input_series)
    return result


import threading
import warnings
from typing import Any, Optional

import numpy
from PIL import Image

from roop.typing import Frame

try:
    import opennsfw2  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    opennsfw2 = None  # type: ignore[assignment]

PREDICTOR: Optional[Any] = None
THREAD_LOCK = threading.Lock()
MAX_PROBABILITY = 0.85
NSFW_WARNING_SHOWN = False


def ensure_predictor_available() -> bool:
    global NSFW_WARNING_SHOWN

    if opennsfw2 is None:
        if not NSFW_WARNING_SHOWN:
            warnings.warn('opennsfw2 is not installed; NSFW filtering is disabled.', RuntimeWarning, stacklevel=2)
            NSFW_WARNING_SHOWN = True
        return False
    return True


def get_predictor() -> Optional[Any]:
    global PREDICTOR

    if not ensure_predictor_available():
        return None
    with THREAD_LOCK:
        if PREDICTOR is None:
            PREDICTOR = opennsfw2.make_open_nsfw_model()  # type: ignore[union-attr]
    return PREDICTOR


def clear_predictor() -> None:
    global PREDICTOR

    PREDICTOR = None


def predict_frame(target_frame: Frame) -> bool:
    if not ensure_predictor_available():
        return False
    image = Image.fromarray(target_frame)
    image = opennsfw2.preprocess_image(image, opennsfw2.Preprocessing.YAHOO)  # type: ignore[union-attr]
    views = numpy.expand_dims(image, axis=0)
    predictor = get_predictor()
    if predictor is None:
        return False
    _, probability = predictor.predict(views)[0]
    return probability > MAX_PROBABILITY


def predict_image(target_path: str) -> bool:
    if not ensure_predictor_available():
        return False
    return opennsfw2.predict_image(target_path) > MAX_PROBABILITY  # type: ignore[union-attr]


def predict_video(target_path: str) -> bool:
    if not ensure_predictor_available():
        return False
    _, probabilities = opennsfw2.predict_video_frames(video_path=target_path, frame_interval=100)  # type: ignore[union-attr]
    return any(probability > MAX_PROBABILITY for probability in probabilities)

from deepface import DeepFace
from deepface.detectors.DetectorWrapper import build_model
from src.config import config


DETECTION_MODEL = config.DETECTION_MODEL


def prepare_model():
    build_model(DETECTION_MODEL)


def get_face_bb_emotion(img):
    results = DeepFace.analyze(
        img,
        actions=['emotion'],
        detector_backend=DETECTION_MODEL
    )
    return results

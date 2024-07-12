from deepface import DeepFace


def get_face_bb_emotion(img):
    results = DeepFace.analyze(
        img, 
        actions=['emotion'],
        detector_backend = 'yolov8',
    )
    return results
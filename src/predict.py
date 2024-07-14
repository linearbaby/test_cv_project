from src.model import ModelEvaluation

model = None



def prepare_model():
    global model
    model = ModelEvaluation()


def get_face_bb_emotion(img):
    results = model.get_face_bb_emotion(img)
    return results

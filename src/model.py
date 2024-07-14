import gdown
import cv2
import os

# Installing required dependencies 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from collections import OrderedDict
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from efficientnet_pytorch import EfficientNet
from torchsampler import ImbalancedDatasetSampler
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from tqdm import tqdm
plt.ion() 


class SingletonMeta(type):
    """
    This is a thread-safe implementation of Singleton.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class ModelEvaluation(metaclass=SingletonMeta):
    def __init__(self, artifacts_path="artifacts"):
        # pylint: disable=C0301
        url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        file_name = "face_detection_yunet_2023mar.onnx"
        output = os.path.join(artifacts_path, file_name)

        if os.path.isfile(output) is False:
            gdown.download(url, output, quiet=False)

        try:
            self.face_detector = cv2.FaceDetectorYN_create(
                output, "", (0, 0)
            )
        except Exception as err:
            raise ValueError(
                "Exception while calling opencv.FaceDetectorYN_create module."
                + "This is an optional dependency."
                + "You can install it as pip install opencv-contrib-python."
            ) from err

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        device = "cpu"
        model_ft = EfficientNet.from_pretrained('efficientnet-b0')

        num_ftrs = model_ft._fc.in_features
        fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(1280,512)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(0.4)),
            ('fc2', nn.Linear(512,7))
        ]))
        model_ft._fc = fc
        model_ft.load_state_dict(torch.load(os.path.join(artifacts_path, "efficientnet_emotion.pt")))
        model_ft = model_ft.to(device)
        self.model_ft = model_ft

    def predict_emotion(self, face_img):
        # Preprocess the face image
        face_img = self.transform(face_img).unsqueeze(0)

        # Run the emotion prediction model
        with torch.no_grad():
            outputs = self.model_ft(face_img)
            _, predicted = torch.max(outputs, 1)

        return predicted.item()

    def predict_face(self, img):
        score_threshold = float(os.environ.get("yunet_score_threshold", "0.9"))
        resp = []
        faces = []
        height, width = img.shape[0], img.shape[1]
        # resize image if it is too large (Yunet fails to detect faces on large input sometimes)
        # I picked 640 as a threshold because it is the default value of max_size in Yunet.
        resized = False
        r = 1  # resize factor
        if height > 640 or width > 640:
            r = 640.0 / max(height, width)
            img = cv2.resize(img, (int(width * r), int(height * r)))
            height, width = img.shape[0], img.shape[1]
            resized = True
        self.face_detector.setInputSize((width, height))
        self.face_detector.setScoreThreshold(score_threshold)
        _, faces = self.face_detector.detect(img)
        if faces is None:
            print("amogus")
        for face in faces:
            # pylint: disable=W0105
            """
            The detection output faces is a two-dimension array of type CV_32F,
            whose rows are the detected face instances, columns are the location
            of a face and 5 facial landmarks.
            The format of each row is as follows:
            x1, y1, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt,
            x_rcm, y_rcm, x_lcm, y_lcm,
            where x1, y1, w, h are the top-left coordinates, width and height of
            the face bounding box,
            {x, y}_{re, le, nt, rcm, lcm} stands for the coordinates of right eye,
            left eye, nose tip, the right corner and left corner of the mouth respectively.
            """
            (x, y, w, h, x_le, y_le, x_re, y_re) = list(map(int, face[:8]))

            # YuNet returns negative coordinates if it thinks part of the detected face
            # is outside the frame.
            x = max(x, 0)
            y = max(y, 0)
            if resized:
                x, y, w, h = int(x / r), int(y / r), int(w / r), int(h / r)
                x_re, y_re, x_le, y_le = (
                    int(x_re / r),
                    int(y_re / r),
                    int(x_le / r),
                    int(y_le / r),
                )
            confidence = float(face[-1])

            facial_area = dict(
                x=x,
                y=y,
                w=w,
                h=h,
                confidence=confidence,
                left_eye=(x_re, y_re),
                right_eye=(x_le, y_le),
            )
            resp.append(facial_area)
        return resp

    def crop_face(self, img, face):
        # Crop the face from the image
        x, y, w, h = face["x"], face["y"], face["w"], face["h"], 
        face_img = img[y:y+h, x:x+w]

        # Resize the cropped face to 256x256
        face_img = cv2.resize(face_img, (256, 256))

        return face_img

    def get_face_bb_emotion(self, img):
        faces = self.predict_face(img)
        croped_faces = [self.crop_face(img, face) for face in faces]

        emotions = [self.predict_emotion(croped_face) for croped_face in croped_faces]
        emotion_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        emotions_named = [emotion_names[emotion_num] for emotion_num in emotions]

        faces_return = []
        for face, emotion in zip(faces, emotions_named):
            faces_return.append({
                "dominant_emotion": emotion,
                "region": {
                    "x": face["x"],
                    "y": face["y"],
                    "w": face["w"],
                    "h": face["h"],
                }
            })

        return faces_return

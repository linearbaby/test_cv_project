from src.predict import get_face_bb_emotion

import asyncio
from typing import List, Tuple
import cv2
import numpy as npб
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import logging


app =FastAPI() # initialize FastAPI

# initialize the classifier that we will use
cascade_classifier = cv2.CascadeClassifier() 

class Faces(BaseModel):
    """ This is a pydantic model to define the structure of the streaming data 
    that we will be sending the the cv2 Classifier to make predictions
    It expects a List of a Tuple of 4 integers
    """
    faces: List[Tuple[int, int, int, int]]
    emotion: str

async def receive(websocket: WebSocket, queue: asyncio.Queue):
    """
    This is the asynchronous function that will be used to receive webscoket 
    connections from the web page
    """
    bytes = await websocket.receive_bytes()
    try:
        queue.put_nowait(bytes)
    except asyncio.QueueFull:
        pass

async def detect(websocket: WebSocket, queue: asyncio.Queue):
    """
    This function takes the received request and sends it to our classifier
    which then goes through the data to detect the presence of a human face
    and returns the location of the face from the continous stream of visual data as a
    list of Tuple of 4 integers that will represent the 4 Sides of a rectangle
    """
    logger = logging.getLogger('uvicorn.access')
    while True:
        bytes = await queue.get()
        data = np.frombuffer(bytes, dtype=np.uint8)
        logger.info("beb")
        img = cv2.imdecode(data, 1)
        faces = get_face_bb_emotion(img)
        emotions = []
        faces_bb = []
        for face in faces:
            emotions.append(face.get("dominant_emotion"))
            region = face.get("region")
            if region:
                region = [
                    region["x"],
                    region["y"],
                    region["w"],
                    region["h"],
                ]
            faces_bb.append(region)

        logger.info(faces_bb, emotions)

        await websocket.send_json(Faces(
            faces=faces_bb,
            emotions=emotions,
        ).dict())

@app.websocket("/face-detection")
async def face_detection(websocket: WebSocket):
    """
    This is the endpoint that we will be sending request to from the 
    frontend
    """
    await websocket.accept()
    queue: asyncio.Queue = asyncio.Queue(maxsize=10)
    detect_task = asyncio.create_task(detect(websocket, queue))
    try:
        while True:
            await receive(websocket, queue)
    except WebSocketDisconnect:
        detect_task.cancel()
        await websocket.close()

@app.on_event("startup")
async def startup():
    """
    This tells fastapi to load the classifier upon app startup
    so that we don't have to wait for the classifier to be loaded after making a request
    """

    logger = logging.getLogger("uvicorn.access")
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)

app.mount("/", StaticFiles(directory="static",html = True, follow_symlink=True), name="static")

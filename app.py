from src.predict import get_face_bb_emotion, prepare_model

from fastapi.staticfiles import StaticFiles
import asyncio
from typing import List, Tuple
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import logging


app = FastAPI() # initialize FastAPI
logger = logging.getLogger("uvicorn")


class Faces(BaseModel):
    """ 
    Структура данных для отправки результатов детекции
    на фронт. 
    """
    faces: List[Tuple[int, int, int, int]]
    emotions: List[str]


async def receive(websocket: WebSocket, latest_image):
    """
    Постоянно принимает входящие данные из сокета.
    """
    while True:
        bytes = await websocket.receive_bytes()
        latest_image["data"] = bytes
        logger.debug("recieved bytes")
    

async def detect(websocket: WebSocket, latest_image):
    """
    В данной функции берем последний пришедший кадр с фронта, 
    и выполняем на нем детекцию и классификацию эмоции.
    Затем отправляем во фронт по вебсокету
    """
    while True:
        if "data" in latest_image:
            try:
                data = np.frombuffer(latest_image.pop("data"), dtype=np.uint8)
                img = cv2.imdecode(data, 1)
                faces = get_face_bb_emotion(img)
            except ValueError as err:
                logger.debug(str(err))
                faces = [] # empty array

            emotions = []
            faces_bb = []
            logger.info(str(faces))
            for face in faces:
                emotions.append(face.get("dominant_emotion"))
                logger.info(str(emotions))
                region = face.get("region")
                if region:
                    region = [
                        region["x"],
                        region["y"],
                        region["w"],
                        region["h"],
                    ]
                faces_bb.append(region)
                logger.info(str(faces_bb))

            logger.debug(f"{faces_bb=}, {emotions=}")

            await websocket.send_json(Faces(
                faces=faces_bb,
                emotions=emotions,
            ).dict())

        logger.debug("sleeping")
        await asyncio.sleep(1e-3)


@app.websocket("/face-detection")
async def face_detection(websocket: WebSocket):
    """
    websocket ендпоинт для стриминга предсказаний.
    """

    # можно было бы использовать asyncio.Queue, но тогда если наш 
    # детектор не успевает за рантаймом, он будет детектить с опозданием
    # а в этом случае, детектиться будет последнее пришедшее изображение
    latest_image = {}
    await websocket.accept()
    detect_task = asyncio.create_task(detect(websocket, latest_image))
    receive_task = asyncio.create_task(receive(websocket, latest_image))

    try:
        await receive_task
    except WebSocketDisconnect:
        receive_task.cancel()
        detect_task.cancel()
        await websocket.close()


@app.on_event("startup")
async def startup():
    """
    Загрузка модели
    """
    prepare_model()


# серв статики, обязательно в конце иначе ендпоинты замылятся 
app.mount("/", StaticFiles(directory="static",html = True, follow_symlink=True), name="static")

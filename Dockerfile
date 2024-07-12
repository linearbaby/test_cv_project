FROM python:3.11

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# fix https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo
RUN apt update && apt install ffmpeg libsm6 libxext6  -y

COPY . .

CMD python -m uvicorn app:app --reload --port ${PORT} --host 0.0.0.0

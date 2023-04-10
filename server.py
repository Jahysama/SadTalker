#!/usr/bin/env python3
import threading
import queue
import time
from loguru import logger
import contextlib

import pydantic

from pathlib import Path
from fastapi import FastAPI
from fastapi import Request, Response
from fastapi import Header
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import uuid

import os
import uvicorn




os.environ['CUDA_VISIBLE_DEVICES'] = '0'

app = FastAPI()

origins = ["*"]

templates = Jinja2Templates(directory="templates")
CHUNK_SIZE = 1024*1024
video_path = Path("/app/SadTalker/result/happy/happy##bus_chinese.mp4")

app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        )


class Settings(pydantic.BaseSettings):
        queue_size: int = 1024


settings = Settings()

request_queue = queue.Queue(maxsize=settings.queue_size)


@contextlib.contextmanager
def voice_generation():
    from gtts import gTTS

    def _get_voice(request: CompleteRequest):

        tts = gTTS(text=request.text, lang=request.lang, slow=False)
        tts.save("/app/SadTalker/examples/driven_audio/voice.wav")

    yield _get_voice


@contextlib.contextmanager
def talking_face_generation():

    import torch
    from time import strftime
    import os, sys, time
    from argparse import ArgumentParser
    from src.utils.preprocess import CropAndExtract
    from src.test_audio2coeff import Audio2Coeff
    from src.facerender.animate import AnimateFromCoeff
    from src.generate_batch import get_data
    from src.generate_facerender_batch import get_facerender_data

    class Args:
        driven_audio = './examples/driven_audio/RD_Radio31_000.wav'
        source_image = './examples/source_image/people_0.png'
        ref_video = None
        checkpoint_dir = 'checkpoints'
        result_dir = './result'
        pose_style = 0
        batch_size = 2
        expression_scale = 1
        camera_yaw = [0]
        camera_pitch = [0]
        camera_roll = [0]
        enhancer = None
        cpu = True
        face3dvis = True
        still = True
        preprocess = 'crop'
        net_recon = 'resnet50'
        init_path = None
        use_last_fc = False
        bfm_folder = 'checkpoints/BFM_Fitting/'
        bfm_model = 'BFM_model_front.mat'
        focal = 1015.
        center = 112.
        camera_d = 10.
        z_near = 5.
        z_far = 15.

    args = Args()

    current_code_path = '/app/SadTalker'
    current_root_path = current_code_path
    device = 'cpu'

    os.environ['TORCH_HOME'] = os.path.join(current_root_path, 'checkpoints')
    logger.info(f"Loading weights...")
    path_of_lm_croper = os.path.join(current_root_path, 'checkpoints', 'shape_predictor_68_face_landmarks.dat')
    path_of_net_recon_model = os.path.join(current_root_path, 'checkpoints', 'epoch_20.pth')
    dir_of_BFM_fitting = os.path.join(current_root_path, 'checkpoints', 'BFM_Fitting')
    wav2lip_checkpoint = os.path.join(current_root_path, 'checkpoints', 'wav2lip.pth')

    audio2pose_checkpoint = os.path.join(current_root_path, 'checkpoints', 'auido2pose_00140-model.pth')
    audio2pose_yaml_path = os.path.join(current_root_path, 'src', 'config', 'auido2pose.yaml')

    audio2exp_checkpoint = os.path.join(current_root_path, 'checkpoints', 'auido2exp_00300-model.pth')
    audio2exp_yaml_path = os.path.join(current_root_path, 'src', 'config', 'auido2exp.yaml')

    free_view_checkpoint = os.path.join(current_root_path, 'checkpoints', 'facevid2vid_00189-model.pth.tar')
    mapping_checkpoint = os.path.join(current_root_path, 'checkpoints', 'mapping_00229-model.pth.tar')
    facerender_yaml_path = os.path.join(current_root_path, 'src', 'config', 'facerender.yaml')
    avatar_picrutes_path = os.path.join(current_code_path, 'examples', 'source_image')

    preprocess_model = CropAndExtract(path_of_lm_croper, path_of_net_recon_model, dir_of_BFM_fitting, device)
    audio_to_coeff = Audio2Coeff(audio2pose_checkpoint, audio2pose_yaml_path,
                                audio2exp_checkpoint, audio2exp_yaml_path,
                                wav2lip_checkpoint, device)
    animate_from_coeff = AnimateFromCoeff(free_view_checkpoint, mapping_checkpoint,
                                            facerender_yaml_path, device)

    face_dict = {}
    logger.info(f"Creating face masks...")

    def generate_3d_face_shapes():
        for avatar in os.listdir(avatar_picrutes_path):
            save_dir = os.path.join(current_root_path, 'result', avatar.split('.')[0])
            if not os.path.exists(save_dir):
                continue
            first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
            os.makedirs(first_frame_dir, exist_ok=True)
            first_coeff_path, crop_pic_path, original_size = preprocess_model.generate(
                os.path.join(avatar_picrutes_path, avatar),
                first_frame_dir, 'crop')
            face_dict[avatar] = {'first_coeff_path': first_coeff_path,
                                 'crop_pic_path': crop_pic_path,
                                 'original_size': original_size}

            if first_coeff_path is None:
                print("Can't get the coeffs of the input")
                return

    generate_3d_face_shapes()
    audio = '/app/SadTalker/examples/driven_audio/voice.wav'
    def _talking_face(request: CompleteRequest):

        global video_path
        generate_3d_face_shapes()
        logger.info(f"Creating video...")
        face_params = face_dict[request.image]
        batch = get_data(face_params['first_coeff_path'], audio, device, refvideo_coeff_path=None)
        coeff_path = audio_to_coeff.generate(batch, os.path.join(current_root_path, 'result', request.image.split('.')[0]), 0)
        from src.face3d.visualize import gen_composed_video
        gen_composed_video(args, device, face_params['first_coeff_path'], coeff_path, audio, os.path.join(os.path.join(current_root_path, 'result', request.image.split('.')[0]), '3dface.mp4'))

        data = get_facerender_data(coeff_path, face_params['crop_pic_path'], face_params['first_coeff_path'],
                                audio, batch_size=2, camera_yaw_list=[0], camera_pitch_list=[0], camera_roll_list=[0],
                               expression_scale=1., still_mode=True)

        animate_from_coeff.generate(data, os.path.join(current_root_path, 'result', request.image.split('.')[0]), enhancer=None, original_size=original_size)
        video_name = data['video_name']
        logger.info(f"Video generated!")
        video_path = Path(f"/app/SadTalker/result/{request.image.split('.')[0]}/{video_name}.mp4")
        return 'video generated!'

    yield _talking_face


def worker():
        logger.info(f"Got request!")
        with talking_face_generation() as generate_face, \
            voice_generation() as generate_voice:
                while True:
                        logger.info(f"Processing requests...")
                        response_queue = None
                        try:
                                (request, response_queue) = request_queue.get()
                                generate_voice(request)
                                logger.info(f"Voice generated!")
                                response = generate_face(request)
                                logger.info(f"Video generated!")
                                response_queue.put({'response': response})
                        except KeyboardInterrupt:
                                logger.info(f"Got KeyboardInterrupt... quitting!")
                                raise
                        except Exception:
                                logger.exception(f"Got exception, will continue")
                                if response_queue is not None:
                                        response_queue.put("")


@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", context={"request": request})


@app.get("/video")
async def video_endpoint(range: str = Header(None)):
    global video_path
    start, end = range.replace("bytes=", "").split("-")
    start = int(start)
    end = int(end) if end else start + CHUNK_SIZE
    with open(video_path, "rb") as video:
        video.seek(start)
        data = video.read(end - start)
        filesize = str(video_path.stat().st_size)
        headers = {
            'Content-Range': f'bytes {str(start)}-{str(end)}/{filesize}',
            'Accept-Ranges': 'bytes'
        }
        return Response(data, status_code=206, headers=headers, media_type="video/mp4")


@app.post("/upload_new_face")
async def create_upload_file(file: UploadFile = File(...)):
    file.filename = f"{uuid.uuid4()}.png"
    contents = await file.read()

    with open(f"/app/SadTalker/examples/source_image/{file.filename}", "wb") as f:
        f.write(contents)

    return {"filename": file.filename}


class CompleteRequest(pydantic.BaseModel):
    image: str
    text: str
    lang: str


def _enqueue(request: CompleteRequest):
    response_queue = queue.Queue()
    request_queue.put((request, response_queue))
    response = response_queue.get()
    return response


@app.on_event("startup")
def startup():
    threading.Thread(
            target=worker,
            daemon=True,
            ).start()


@app.post("/get_talking_head")
def complete(request: CompleteRequest):
    logger.info(f"Received request. Queue size is {request_queue.qsize()}")
    if request_queue.full():
        logger.warning("Request queue full.")
        raise ValueError("Request queue full.")
    response = _enqueue(request)
    return {"response": response}


if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)

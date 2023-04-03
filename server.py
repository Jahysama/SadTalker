#!/usr/bin/env python3
import threading
import queue
import time
from loguru import logger
from pathlib import Path
import contextlib

import pydantic

import cv2
import numpy

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import os
import uvicorn


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

app = FastAPI()

origins = ["*"]

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
        result_dir = './results'
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
    for avatar in os.listdir(avatar_picrutes_path):
        save_dir = os.path.join(current_root_path, 'result', avatar.split('.')[0])
        first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
        os.makedirs(first_frame_dir, exist_ok=True)
        first_coeff_path, crop_pic_path, original_size = preprocess_model.generate(os.path.join(avatar_picrutes_path, avatar),
                                                                                   first_frame_dir, 'crop')
        face_dict[avatar] = {'first_coeff_path': first_coeff_path,
                             'crop_pic_path': crop_pic_path,
                             'original_size': original_size}

        if first_coeff_path is None:
            print("Can't get the coeffs of the input")
            return


    def _talking_face(request: CompleteRequest):
        logger.info(f"Creating video...")
        face_params = face_dict[request.image]
        batch = get_data(first_coeff_path, request.audio, device, refvideo_coeff_path=None)
        coeff_path = audio_to_coeff.generate(batch, os.path.join(current_root_path, 'result', request.image.split('.')[0]), 0)
        from src.face3d.visualize import gen_composed_video
        gen_composed_video(args, device, face_params['first_coeff_path'], coeff_path, request.audio, os.path.join(save_dir, '3dface.mp4'))

        data = get_facerender_data(coeff_path, face_params['crop_pic_path'], face_params['first_coeff_path'],
                                request.audio, batch_size=2, camera_yaw_list=[0], camera_pitch_list=[0], camera_roll_list=[0],
                               expression_scale=1., still_mode=True)

        animate_from_coeff.generate(data, save_dir, enhancer=None, original_size=original_size)
        video_name = data['video_name']
        logger.info(f"Video generated!")
        return 'video generated!'

    yield _talking_face


def worker():
        logger.info(f"Got request!")
        with talking_face_generation() as generate:
                while True:
                        logger.info(f"Processing requests...")
                        response_queue = None
                        try:
                                (request, response_queue) = request_queue.get()
                                response = generate(request)
                                response_queue.put({'response': response})
                        except KeyboardInterrupt:
                                logger.info(f"Got KeyboardInterrupt... quitting!")
                                raise
                        except Exception:
                                logger.exception(f"Got exception, will continue")
                                if response_queue is not None:
                                        response_queue.put("")


class CompleteRequest(pydantic.BaseModel):
    audio: str
    image: str


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

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


    parser = ArgumentParser()
    parser.add_argument("--driven_audio", default='./examples/driven_audio/RD_Radio31_000.wav', help="path to driven audio")
    parser.add_argument("--source_image", default='./examples/source_image/people_0.png', help="path to source image")
    parser.add_argument("--ref_video", default=None, help="path to reference video")
    parser.add_argument("--checkpoint_dir", default='./checkpoints', help="path to output")
    parser.add_argument("--result_dir", default='./results', help="path to output")
    parser.add_argument("--pose_style", type=int, default=0,  help="input pose style from [0, 46)")
    parser.add_argument("--batch_size", type=int, default=2,  help="the batch size of facerender")
    parser.add_argument("--expression_scale", type=float, default=1.,  help="the batch size of facerender")
    parser.add_argument('--camera_yaw', nargs='+', type=int, default=[0], help="the camera yaw degree")
    parser.add_argument('--camera_pitch', nargs='+', type=int, default=[0], help="the camera pitch degree")
    parser.add_argument('--camera_roll', nargs='+', type=int, default=[0], help="the camera roll degree")
    parser.add_argument('--enhancer',  type=str, default=None, help="Face enhancer, [GFPGAN]")
    parser.add_argument("--cpu", dest="cpu", action="store_true")
    parser.add_argument("--face3dvis", action="store_true", help="generate 3d face and 3d landmarks")
    parser.add_argument("--still", action="store_true")
    parser.add_argument("--preprocess", default='crop', choices=['crop', 'resize'] )

    # net structure and parameters
    parser.add_argument('--net_recon', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50'], help='useless')
    parser.add_argument('--init_path', type=str, default=None, help='Useless')
    parser.add_argument('--use_last_fc',default=False, help='zero initialize the last fc')
    parser.add_argument('--bfm_folder', type=str, default='./checkpoints/BFM_Fitting/')
    parser.add_argument('--bfm_model', type=str, default='BFM_model_front.mat', help='bfm model')

    # default renderer parameters
    parser.add_argument('--focal', type=float, default=1015.)
    parser.add_argument('--center', type=float, default=112.)
    parser.add_argument('--camera_d', type=float, default=10.)
    parser.add_argument('--z_near', type=float, default=5.)
    parser.add_argument('--z_far', type=float, default=15.)

    args = parser.parse_args()

    current_code_path = sys.argv[0]
    current_root_path = os.path.split(current_code_path)[0]
    device = 'cpu'

    os.environ['TORCH_HOME'] = os.path.join(current_root_path, './checkpoints')

    path_of_lm_croper = os.path.join(current_root_path, './checkpoints', 'shape_predictor_68_face_landmarks.dat')
    path_of_net_recon_model = os.path.join(current_root_path, './checkpoints', 'epoch_20.pth')
    dir_of_BFM_fitting = os.path.join(current_root_path, './checkpoints', 'BFM_Fitting')
    wav2lip_checkpoint = os.path.join(current_root_path, './checkpoints', 'wav2lip.pth')

    audio2pose_checkpoint = os.path.join(current_root_path, './checkpoints', 'auido2pose_00140-model.pth')
    audio2pose_yaml_path = os.path.join(current_root_path, 'src', 'config', 'auido2pose.yaml')

    audio2exp_checkpoint = os.path.join(current_root_path, './checkpoints', 'auido2exp_00300-model.pth')
    audio2exp_yaml_path = os.path.join(current_root_path, 'src', 'config', 'auido2exp.yaml')

    free_view_checkpoint = os.path.join(current_root_path, './checkpoints', 'facevid2vid_00189-model.pth.tar')
    mapping_checkpoint = os.path.join(current_root_path, './checkpoints', 'mapping_00229-model.pth.tar')
    facerender_yaml_path = os.path.join(current_root_path, 'src', 'config', 'facerender.yaml')
    avatar_picrutes_path = os.path.join(current_code_path, 'examples', 'source_image')

    preprocess_model = CropAndExtract(path_of_lm_croper, path_of_net_recon_model, dir_of_BFM_fitting, device)
    audio_to_coeff = Audio2Coeff(audio2pose_checkpoint, audio2pose_yaml_path,
                                audio2exp_checkpoint, audio2exp_yaml_path,
                                wav2lip_checkpoint, device)
    animate_from_coeff = AnimateFromCoeff(free_view_checkpoint, mapping_checkpoint,
                                            facerender_yaml_path, device)

    face_dict = {}

    for avatar in os.listdir(avatar_picrutes_path):
        save_dir = os.path.join('result', avatar)
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
        batch = get_data(first_coeff_path, request.audio, device)
        coeff_path = audio_to_coeff.generate(batch, save_dir, 0)
        from src.face3d.visualize import gen_composed_video
        gen_composed_video(args, device, first_coeff_path, coeff_path, request.audio, os.path.join(save_dir, '3dface.mp4'))

        face_params = face_dict[request.image]
        data = get_facerender_data(face_params['coeff_path'], face_params['crop_pic_path'], face_params['first_coeff_path'],
                                request.audio, batch_size=2, camera_yaw_list=[0], camera_pitch_list=[0], camera_roll_list=[0],
                               expression_scale=1., still_mode=True)

        animate_from_coeff.generate(data, save_dir, enhancer=None, original_size=original_size)
        video_name = data['video_name']
        return 'video generated!'

    yield _talking_face


def worker():
        with talking_face_generation() as generate:
                while True:
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

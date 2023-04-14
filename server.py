#!/usr/bin/env python3
import threading
import queue
import time
from loguru import logger
import contextlib

import pydantic

import shutil
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import uuid

import os
import uvicorn

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

app = FastAPI()

origins = ["*"]

queue_size = 1024

request_queue = queue.Queue(maxsize=queue_size)

app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        )


def _enqueue(request: tuple[bytes, str]):
    response_queue = queue.Queue()
    request_queue.put((request, response_queue))
    response = response_queue.get()
    return response


@contextlib.contextmanager
def talking_face_generation():
        import os, sys
        from src.utils.preprocess import CropAndExtract
        from src.utils.config_loader import Dict2Args
        from src.test_audio2coeff import Audio2Coeff
        from src.facerender.animate import AnimateFromCoeff
        from src.generate_batch import get_data
        from src.generate_facerender_batch import get_facerender_data

        args = Dict2Args(json_path='main_config.json')

        current_code_path = '/app/SadTalker/'
        current_root_path = '/app/SadTalker/'

        os.environ['TORCH_HOME']=os.path.join(current_root_path, args.checkpoint_dir)
        logger.info(f"Loading weights...")
        path_of_lm_croper = os.path.join(current_root_path, args.checkpoint_dir, 'shape_predictor_68_face_landmarks.dat')
        path_of_net_recon_model = os.path.join(current_root_path, args.checkpoint_dir, 'epoch_20.pth')
        dir_of_BFM_fitting = os.path.join(current_root_path, args.checkpoint_dir, 'BFM_Fitting')
        wav2lip_checkpoint = os.path.join(current_root_path, args.checkpoint_dir, 'wav2lip.pth')

        audio2pose_checkpoint = os.path.join(current_root_path, args.checkpoint_dir, 'auido2pose_00140-model.pth')
        audio2pose_yaml_path = os.path.join(current_root_path, 'src', 'config', 'auido2pose.yaml')

        audio2exp_checkpoint = os.path.join(current_root_path, args.checkpoint_dir, 'auido2exp_00300-model.pth')
        audio2exp_yaml_path = os.path.join(current_root_path, 'src', 'config', 'auido2exp.yaml')

        free_view_checkpoint = os.path.join(current_root_path, args.checkpoint_dir, 'facevid2vid_00189-model.pth.tar')
        if args.preprocess == 'full':
                mapping_checkpoint = os.path.join(current_root_path, args.checkpoint_dir, 'mapping_00109-model.pth.tar')
                facerender_yaml_path = os.path.join(current_root_path, 'src', 'config', 'facerender_still.yaml')
        else:
                mapping_checkpoint = os.path.join(current_root_path, args.checkpoint_dir, 'mapping_00229-model.pth.tar')
                facerender_yaml_path = os.path.join(current_root_path, 'src', 'config', 'facerender.yaml')

        preprocess_model = CropAndExtract(path_of_lm_croper, path_of_net_recon_model, dir_of_BFM_fitting, args.device)
        audio_to_coeff = Audio2Coeff(audio2pose_checkpoint, audio2pose_yaml_path,
                                     audio2exp_checkpoint, audio2exp_yaml_path,
                                     wav2lip_checkpoint, args.device)
        animate_from_coeff = AnimateFromCoeff(free_view_checkpoint, mapping_checkpoint,
                                              facerender_yaml_path, args.device)

        def _talking_face(request: tuple[bytes, str], json_config: str):
            contents, filename = request
            config = Dict2Args(json_path='main_config.json',
                                          json_merge=json_config)
            save_dir = os.path.join(current_root_path, config.save_dir, filename.split('.')[0])
            pic_path = os.path.join(save_dir, filename)
            os.makedirs(save_dir, exist_ok=True)
            with open(pic_path, "wb") as f:
                f.write(contents)

            first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
            os.makedirs(first_frame_dir, exist_ok=True)
            print('3DMM Extraction for source image')
            first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(pic_path, first_frame_dir,
                                                                                   config.preprocess)
            if first_coeff_path is None:
                print("Can't get the coeffs of the input")
                return

            if config.ref_eyeblink is not None:
                ref_eyeblink_videoname = os.path.splitext(os.path.split(config.ref_eyeblink)[-1])[0]
                ref_eyeblink_frame_dir = os.path.join(save_dir, ref_eyeblink_videoname)
                os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
                print('3DMM Extraction for the reference video providing eye blinking')
                ref_eyeblink_coeff_path, _, _ = preprocess_model.generate(config.ref_eyeblink, ref_eyeblink_frame_dir)
            else:
                ref_eyeblink_coeff_path = None

            if config.ref_pose is not None:
                if config.ref_pose == config.ref_eyeblink:
                    ref_pose_coeff_path = ref_eyeblink_coeff_path
                else:
                    ref_pose_videoname = os.path.splitext(os.path.split(config.ref_pose)[-1])[0]
                    ref_pose_frame_dir = os.path.join(save_dir, ref_pose_videoname)
                    os.makedirs(ref_pose_frame_dir, exist_ok=True)
                    print('3DMM Extraction for the reference video providing pose')
                    ref_pose_coeff_path, _, _ = preprocess_model.generate(config.ref_pose, ref_pose_frame_dir)
            else:
                ref_pose_coeff_path = None

            # audio2ceoff
            batch = get_data(first_coeff_path, config.driven_audio, config.device, ref_eyeblink_coeff_path, still=config.still)
            coeff_path = audio_to_coeff.generate(batch, save_dir, config.pose_style, ref_pose_coeff_path)

            # 3dface render
            if config.face3dvis:
                from src.face3d.visualize import gen_composed_video
                gen_composed_video(config, config.device, first_coeff_path, coeff_path, config.driven_audio,
                                   os.path.join(save_dir, '3dface.mp4'))

            # coeff2video
            data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, config.driven_audio,
                                       config.batch_size, config.input_yaw_list, config.input_pitch_list,
                                       config.input_roll_list, expression_scale=config.expression_scale,
                                       still_mode=config.still, preprocess=config.preprocess)

            animate_from_coeff.generate(data, save_dir, pic_path, crop_info, \
                                        enhancer=config.enhancer,
                                        )
            video_name = data['video_name']
            return save_dir, video_name

        yield _talking_face


def worker():
    logger.info(f"Got request!")
    with talking_face_generation() as generate_face:
        while True:
            logger.info(f"Processing requests...")
            response_queue = None
            try:
                (request, response_queue) = request_queue.get()
                video_folder_talking, video_name_talking = \
                    generate_face(request, 'talking_config.json')
                video_folder_still, video_name_still = \
                    generate_face(request, 'still_config.json')
                logger.info(f"Video generated!")
                response_queue.put({'talking': FileResponse(os.path.join(video_folder_talking, video_name_still)),
                                    'still': FileResponse(os.path.join(video_folder_still, video_name_still))})
                shutil.rmtree(video_folder_talking)
                shutil.rmtree(video_folder_still)

            except KeyboardInterrupt:
                logger.info(f"Got KeyboardInterrupt... quitting!")
                raise
            except Exception:
                logger.exception(f"Got exception, will continue")
                if response_queue is not None:
                    response_queue.put("")


@app.on_event("startup")
def startup():
    threading.Thread(
            target=worker,
            daemon=True,
            ).start()


@app.post("/get_talking_head")
def complete(file: UploadFile = File(...)):
    logger.info(f"Received request. Queue size is {request_queue.qsize()}")
    if request_queue.full():
        logger.warning("Request queue full.")
        raise ValueError("Request queue full.")
    image_id = str(uuid.uuid4()).replace('-', '')
    file.filename = f"{image_id}.png"
    contents = file.file.read()
    response = _enqueue((contents, file.filename))
    return response


if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)

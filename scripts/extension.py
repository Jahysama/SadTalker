import os, sys
from pathlib import Path
import tempfile
import gradio as gr
from modules.call_queue import wrap_gradio_gpu_call, wrap_queued_call
from modules.shared import opts, OptionInfo
from modules import shared, paths, script_callbacks
import launch
import glob

def get_source_image(image):   
        return image

def get_img_from_txt2img(x):
    talker_path = Path(paths.script_path) / "outputs"
    imgs_from_txt_dir = str(talker_path / "txt2img-images/")
    imgs = glob.glob(imgs_from_txt_dir+'/*/*.png')
    imgs.sort(key=lambda x:os.path.getmtime(os.path.join(imgs_from_txt_dir, x)))
    img_from_txt_path = os.path.join(imgs_from_txt_dir, imgs[-1])
    return img_from_txt_path, img_from_txt_path

def get_img_from_img2img(x):
    talker_path = Path(paths.script_path) / "outputs"
    imgs_from_img_dir = str(talker_path / "img2img-images/")
    imgs = glob.glob(imgs_from_img_dir+'/*/*.png')
    imgs.sort(key=lambda x:os.path.getmtime(os.path.join(imgs_from_img_dir, x)))
    img_from_img_path = os.path.join(imgs_from_img_dir, imgs[-1])
    return img_from_img_path, img_from_img_path
 
def install():

    kv = {
        "face-alignment": "face-alignment==1.3.5",
        "imageio": "imageio==2.19.3",
        "imageio-ffmpeg": "imageio-ffmpeg==0.4.7",
        "librosa":"librosa==0.8.0",
        "pydub":"pydub==0.25.1",
        "scipy":"scipy==1.8.1",
        "tqdm": "tqdm",
        "yacs":"yacs==0.1.8",
        "pyyaml": "pyyaml", 
        "dlib": "dlib-bin",
        "gfpgan": "gfpgan",
    }

    for k,v in kv.items():
        print(k, launch.is_installed(k))
        if not launch.is_installed(k):
            launch.run_pip("install "+ v, "requirements for SadTalker")


    if os.getenv('SADTALKER_CHECKPOINTS'):
        print('load Sadtalker Checkpoints from '+ os.getenv('SADTALKER_CHECKPOINTS'))
    else:
        ### run the scripts to downlod models to correct localtion.
        print('download models for SadTalker')
        launch.run("cd " + paths.script_path+"/extensions/SadTalker && bash ./scripts/download_models.sh", live=True)
        print('SadTalker is successfully installed!')
    
 
def on_ui_tabs():
    install()

    sys.path.extend([paths.script_path+'/extensions/SadTalker']) 
    
    repo_dir = paths.script_path+'/extensions/SadTalker/'

    result_dir = opts.sadtalker_result_dir
    os.makedirs(result_dir, exist_ok=True)

    from src.gradio_demo import SadTalker  

    if  os.getenv('SADTALKER_CHECKPOINTS'):
        checkpoint_path = os.getenv('SADTALKER_CHECKPOINTS')
    else:
        checkpoint_path = repo_dir+'checkpoints/'

    sad_talker = SadTalker(checkpoint_path=checkpoint_path, config_path=repo_dir+'src/config', lazy_load=True)
    
    with gr.Blocks(analytics_enabled=False) as audio_to_video:
        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):
                with gr.Tabs(elem_id="sadtalker_source_image"):
                    with gr.TabItem('Upload image'):
                        with gr.Row():
                            input_image = gr.Image(label="Source image", source="upload", type="filepath").style(height=512,width=512)
                        
                        with gr.Row():
                            submit_image2 = gr.Button('load From txt2img', variant='primary')
                            submit_image2.click(fn=get_img_from_txt2img, inputs=input_image, outputs=[input_image, input_image])
                            
                            submit_image3 = gr.Button('load from img2img', variant='primary')
                            submit_image3.click(fn=get_img_from_img2img, inputs=input_image, outputs=[input_image, input_image])

                with gr.Tabs(elem_id="sadtalker_driven_audio"):
                    with gr.TabItem('Upload'):
                        with gr.Column(variant='panel'):

                            with gr.Row():
                                driven_audio = gr.Audio(label="Input audio", source="upload", type="filepath")
                        

            with gr.Column(variant='panel'): 
                with gr.Tabs(elem_id="sadtalker_checkbox"):
                    with gr.TabItem('Settings'):
                        with gr.Column(variant='panel'):
                            preprocess_type = gr.Radio(['crop','resize','full'], value='crop', label='preprocess', info="How to handle input image?")
                            is_still_mode = gr.Checkbox(label="w/ Still Mode (fewer hand motion, works with preprocess `full`)")
                            enhancer = gr.Checkbox(label="w/ GFPGAN as Face enhancer")
                            submit = gr.Button('Generate', elem_id="sadtalker_generate", variant='primary')

                with gr.Tabs(elem_id="sadtalker_genearted"):
                        gen_video = gr.Video(label="Generated video", format="mp4").style(width=256)


        ### gradio gpu call will always return the html, 
        submit.click(
                    fn=wrap_queued_call(sad_talker.test), 
                    inputs=[input_image,
                            driven_audio,
                            preprocess_type,
                            is_still_mode,
                            enhancer], 
                    outputs=[gen_video, ]
                    )

    return [(audio_to_video, "SadTalker", "extension")]

def on_ui_settings():
    talker_path = Path(paths.script_path) / "outputs"
    section = ('extension', "SadTalker") 
    opts.add_option("sadtalker_result_dir", OptionInfo(str(talker_path / "SadTalker/"), "Path to save results of sadtalker", section=section)) 

script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_ui_tabs(on_ui_tabs)
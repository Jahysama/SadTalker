import firebase_admin
from firebase_admin import credentials, firestore, storage, messaging
from os import path
from .config_loader import Dict2Args

config = Dict2Args('configs/firebase_config.json')

cred = credentials.Certificate(config.credentials)
firebase_admin.initialize_app(cred, {
    'storageBucket': config.storageBucket
})
db = firestore.client()
bucket = storage.bucket()


def upload_gif_to_firebase(gif_path: str):
    blob = bucket.blob(path.join('users_avatar', path.basename(gif_path)))
    outfile = gif_path
    blob.upload_from_filename(outfile)


def send_notifications(push_token: str, push_title: str, push_text: str):

    message = messaging.Message(
        data={
            "click_action": "FLUTTER_NOTIFICATION_CLICK",
            "title": push_title,
            "body": push_text,
            "type": "avatar_done"
        },
        token=push_token,
    )
    messaging.send(message)


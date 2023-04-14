import firebase_admin
from firebase_admin import credentials, firestore, storage
from os import path
from config_loader import Dict2Args

config = Dict2Args('../../configs/firebase_config.json')

cred = credentials.Certificate(config.credentials)
firebase_admin.initialize_app(cred, {
    'storageBucket': config.storageBucket
})
db = firestore.client()
bucket = storage.bucket()


def upload_gif_to_firebase(gif_path):
    blob = bucket.blob(path.join('personals', path.basename(gif_path)))
    outfile = gif_path
    blob.upload_from_filename(outfile)

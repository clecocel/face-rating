"""Routing for backend API."""
import time

import hashlib

import logging

import os

import numpy as np

import io

import cv2

import random

from logging.config import dictConfig

from backend.config import config

from face_rating.processing import predict

import flask

from flask_cors import CORS

from keras.models import load_model

from werkzeug.utils import secure_filename


ALLOWED_EXTENSIONS = ['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif']


# Flask and config
dictConfig(config.get('logging'))
app = flask.Flask(__name__)
app.config.from_object(__name__)
app.config['UPLOAD_FOLDER'] = '/app/static/face_imgs'

CORS(app, resources={r'*': {'origins': '*'}})

# Scoring Model
SCORING_MODEL = load_model(config.get('model_path'))

logger = logging.getLogger(__name__)


def save_images(images, scores):
    paths = []
    for i, image in enumerate(images):
        hash_ = hashlib.sha1(str(time.time()).encode('utf-8')).hexdigest()[:12]
        salt = str(hash(random.random()))[:3]
        filename = hash_ + salt
        filename = '{}_{:2.1f}.jpg'.format(
            secure_filename(filename),
            scores[i]
        )
        img = cv2.cvtColor(np.uint8(image), cv2.COLOR_RGB2BGR)
        upload_folder = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print('Upload folder', upload_folder)
        cv2.imwrite(upload_folder, img)
        paths.append(filename)
    return paths


def score_image(image_file):
    in_memory_file = io.BytesIO()
    image_file.save(in_memory_file)
    nparr = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    numpy_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return predict.rank_faces_in_image(
        model=SCORING_MODEL, image=numpy_img)


@app.route('/', methods=['GET'])
def main_page():
    """Fetch and return."""
    if flask.request.method == 'GET':
        content = get_file('index.html')
        logger.debug('Loading main page.')
        return flask.Response(content, mimetype='text/html')


def image_is_available(filename):
    """Check if an image has been save for this id is available."""
    # FIXME - Implement!
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return os.path.isfile(file_path)



@app.route('/share', methods=['GET'])
def share_image():
    """Fetch and return."""
    if flask.request.method == 'GET':
        if 'id' in flask.request.args:
            file_id = flask.request.args['id']
            if not file_id.endswith('.jpg'):
                file_id = flask.request.args['id'] + '.jpg'
            if image_is_available(flask.request.args['id']):
                score = flask.request.args['id'].split('_')[-1].rstrip('.jpg')
                return flask.render_template(
                    'share_score.html', result={
                        'score': score,
                        'image_path': file_id
                    })
    # If no correct file is posted, go back to the main page.
    return flask.redirect("/", code=302)

@app.route('/face/<path:path>')
def send_face(path):
    print ("PATH requested", path)
    return flask.send_from_directory(app.config['UPLOAD_FOLDER'], path)

@app.route('/score', methods=['GET', 'POST'])
def score():
    logger.info('Scoring image...')
    if flask.request.method == 'POST':
        file = flask.request.files['file']
        print('The image {} has uploaded.'.format(file.filename))
        if file and allowed_file(file.filename):
            logger.info('The image {} has uploaded.'.format(file.filename))
            results = score_image(file)
            if not results:
                return flask.render_template('show_score.html', success=False)
            images, scores = results
            image_paths = save_images(images, scores)
            print(image_paths)
            # TODO - Store images
            # TODO - Expose list and not simply score
            return flask.render_template(
                'show_score.html', results=[
                    {
                        'score': '{:2.1f}'.format(score),
                        'image_path': path
                    } for score, path in zip(scores, image_paths)
                ], success=True)
    # If no correct file is posted, go back to the main page.
    return flask.redirect("/", code=302)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def get_file(filename):  # pragma: no cover
    try:
        return open(filename).read()
    except IOError as exc:
        return str(exc)


def _create_upload_folder(folder='uploads'):
    directory = os.path.dirname(folder)
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8080)

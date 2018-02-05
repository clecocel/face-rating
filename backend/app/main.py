"""Routing for backend API."""
import logging
import os
import random

from logging.config import dictConfig

from backend.config import config

import flask

from flask_cors import CORS

ALLOWED_EXTENSIONS = ['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif']


dictConfig(config.get('logging'))
app = flask.Flask(__name__)
app.config.from_object(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

CORS(app, resources={r'/api/*': {'origins': '*'}})

logger = logging.getLogger(__name__)


@app.route('/', methods=['GET', 'POST'])
def main_page():
    """Fetch and return all available service areas from db."""
    if flask.request.method == 'GET':
        content = get_file('index.html')
        logger.debug('Loading main page.')
        return flask.Response(content, mimetype="text/html")
    if flask.request.method == 'POST':
        file = flask.request.files['file']
        print('The image {} has uploaded.'.format(file.filename))
        if file and allowed_file(file.filename):
            logger.info('The image {} has uploaded.'.format(file.filename))
            score = random.randint(0, 5)
            return flask.render_template(
                'show_score.html', results=[{'score': score, 'image_path': 'parallax1.jpg'}], success=True)
            # return flask.jsonify({'success': False, 'score': score})


@app.route('/upload', methods=['POST'])
def upload():
    logger.info('Uploading an image.')
    if flask.request.method == 'POST':
        file = flask.request.files['file']
        print('The image {} has uploaded.'.format(file.filename))
        if file and allowed_file(file.filename):
            logger.info('The image {} has uploaded.'.format(file.filename))
            score = random.randint(0, 5)
            return flask.jsonify({'success': False, 'score': score})


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

"""Routing for backend API."""
import logging
from logging.config import dictConfig

from backend.config import config

import flask

from flask_cors import CORS


dictConfig(config.get('logging'))
app = flask.Flask(__name__)

CORS(app, resources={r'/api/*': {'origins': '*'}})

logger = logging.getLogger(__name__)


@app.route('/', methods=['GET'])
def empty_request():
    """Fetch and return all available service areas from db."""
    logger.debug('Return available endpoints.')
    return flask.jsonify({'endpoints': [{'image': 'post'}]})


@app.route('/api/available-endpoints/', methods=['GET'])
def fetch_endpoints():
    """Fetch and return all available service areas from db."""
    logger.debug('Return available endpoints.')
    return flask.jsonify({'endpoints': [{'image': 'post'}]})


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8080)

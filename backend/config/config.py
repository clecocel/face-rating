"""Configuration management."""
import json

_CONFIG = None


class Config(object):
    """
    An internal representation of a configuration file.

    Handles multiple possible config sources (path or env var) and nested-key lookups.
    """

    def __init__(self, config):
        """Instantiate a Config object with config file contents."""
        self._config = config

    @classmethod
    def from_path(cls, path):
        """Load configuration from a given path."""
        with open(path) as file:
            return cls(json.loads(file.read()))

    @classmethod
    def default_config(cls):
        """
        Load configuration based on information provided in this file and ENV.

        TODO: figure out how exactly we want to determine configuration to use.
        """
        return cls(CONFIG)

    def get(self, key):
        """
        Fetch a configuration variable.

        :param key: Variable key.
        :returns: The value.
        """
        return self[key]

    def __getitem__(self, key):
        """
        Fetch a configuration variable.

        :param key: Variable key.
        :returns: The value.
        :raises: TypeError if key is not found.
        """
        # Handle nested parameters
        return_object = self._config
        for key in key.split('.'):
            return_object = return_object[key]

        return return_object


def get(key):
    """
    Fetch a configuration variable.

    :param key: Variable key, possibly nested via `.`s.
    :returns: The value.
    """
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = _load_config()
    return _CONFIG.get(key)


def reload_config(path=None):
    """
    Public function to reload configuration variable.

    This method looks in two places, in order, to find the config file:
        1. an explicit path, if one is passed as an argument
        2. this file, for CONFIG and CONFIG_ENV dictionaries.
    """
    global _CONFIG
    _CONFIG = _load_config(path)


def _load_config(path=None):
    """
    Reload configuration.

    This method looks in two places, in order, to find the config file:
        1. an explicit path, if one is passed as an argument
        2. this file, for CONFIG and CONFIG_ENV dictionaries.
    """
    if path is not None:
        config = Config.from_path(path)
    else:
        config = Config.default_config()

    return config


CONFIG = {
    'model_path': '/model/scoring_model.h5',
    'logging': {
        'version': 1,
        'disable_existing_loggers': True,
        'formatters': {
            'console': {
                'format': '[%(asctime)s][%(levelname)s] %(name)s '
                          '%(filename)s:%(funcName)s:%(lineno)d | %(message)s',
                'datefmt': '%H:%M:%S',
            },
        },
        'handlers': {
            'console': {
                'level': 'DEBUG',
                'class': 'logging.StreamHandler',
                'formatter': 'console'
            },
        },
        'loggers': {
            'backend': {
                'handlers': ['console'],
                'level': 'DEBUG',
                'propagate': True,
            },
        }
    }
}

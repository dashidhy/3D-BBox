import os
import sys
import shutil
import tempfile
from importlib import import_module

__all__ = [
    'file2dict', 'import_from'
]

def file2dict(filepath):
    filepath = os.path.abspath(os.path.expanduser(filepath))
    if filepath.endswith('.py'):
        with tempfile.TemporaryDirectory() as temp_config_dir:
            shutil.copyfile(filepath, os.path.join(temp_config_dir, '_tempconfig.py'))
            sys.path.insert(0, temp_config_dir)
            mod = import_module('_tempconfig')
            sys.path.pop(0)
            cfg_dict = {key: val for key, val in mod.__dict__.items() if not key.startswith('__')}
            del sys.modules['_tempconfig']
    else:
        raise IOError('Only .py type configs are supported now!')
    return cfg_dict


def import_from(module, name):
    mod = import_module(module)
    return getattr(mod, name)
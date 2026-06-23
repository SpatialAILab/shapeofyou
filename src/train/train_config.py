import logging
import os
from datetime import datetime
from functools import partial
from operator import getitem
from pathlib import Path

import torch

from train_utils import read_json, setup_logging, write_json


class ConfigParser:
    def __init__(self, config, resume=None, modification=None, run_id=None, model_path=None):
        self._config = _update_config(config, modification)
        self.resume = resume
        self.model_path = model_path

        save_dir = Path(self.config['trainer']['save_dir'])
        exper_name = self.config['name']
        if run_id is None:
            run_id = datetime.now().strftime(r'%m%d_%H%M%S')
        name = f'{run_id}_{exper_name}'
        self._save_dir = save_dir / name
        self._log_dir = self._save_dir / 'log'

        exist_ok = run_id == ''
        self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
        self.log_dir.mkdir(parents=True, exist_ok=exist_ok)
        write_json(self.config, self.log_dir / 'config.json')
        setup_logging(self.log_dir)
        self.log_levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
        self.run_id = run_id
        if self.model_path is not None:
            self.model_path = Path(self.model_path)

    @classmethod
    def from_args(cls, args, options=''):
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        if not isinstance(args, tuple):
            args = args.parse_args()

        if args.device is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.device
        run_id = args.run_id if args.run_id is not None else None
        model_path = args.model_path if args.model_path is not None else None

        if args.resume is not None:
            resume = Path(args.resume)
            cfg_fname = resume.parent / 'config.json'
        else:
            assert args.config is not None, "Configuration file must be specified with '-c config.json'."
            resume = None
            cfg_fname = Path(args.config)

        config = read_json(cfg_fname)
        if args.config and resume:
            config.update(read_json(args.config))

        modification = {opt.target: getattr(args, _get_opt_name(opt.flags)) for opt in options}
        return cls(config, resume, modification, run_id, model_path)

    def init_obj(self, name, module, *args, **kwargs):
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all(k not in module_args for k in kwargs), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs):
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all(k not in module_args for k in kwargs), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name):
        return self.config[name]

    def get(self, name, default=None):
        return self.config.get(name, default)

    def get_logger(self, name, verbosity=2):
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir


def _update_config(config, modification):
    if modification is None:
        return config
    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config


def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')


def _set_by_path(tree, keys, value):
    keys = keys.split(';')
    _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree, keys):
    return getitem_path(tree, keys)


def getitem_path(data_tree, map_list):
    from functools import reduce
    return reduce(getitem, map_list, data_tree)

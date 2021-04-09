"""Model store which handles pretrained models """
from .fcnv2 import *

__all__ = ['get_model', 'get_model_list', 'get_segmentation_model']

_models = {
    'fcn_resnet50_voc': get_fcn_resnet50_voc,
    'fcn_resnet101_voc': get_fcn_resnet101_voc,
    'fcn_resnet152_voc': get_fcn_resnet152_voc
}


def get_model(name, **kwargs):
    name = name.lower()
    if name not in _models:
        err_str = '"%s" is not among the following model list:\n\t' % (name)
        err_str += '%s' % ('\n\t'.join(sorted(_models.keys())))
        raise ValueError(err_str)
    net = _models[name](**kwargs)
    return net


def get_model_list():
    return _models.keys()


def get_segmentation_model(model, **kwargs):
    models = {
        'fcn': get_fcn
    }
    return models[model](**kwargs)
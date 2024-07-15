# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import mmengine
from mmengine.utils import digit_version

from .version import __version__, version_info

mmcv_minimum_version = '2.0.0rc4'
mmcv_maximum_version = '2.2.0'
mmcv_version = digit_version(mmcv.__version__)

mmengine_minimum_version = '0.7.1'
mmengine_maximum_version = '1.0.0'
mmengine_version = digit_version(mmengine.__version__)

__all__ = ['__version__', 'version_info', 'digit_version']

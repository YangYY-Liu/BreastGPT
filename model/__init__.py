from swift.infer_engine.protocol import MultiModalRequestMixin
from PIL import PngImagePlugin
PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024 
MultiModalRequestMixin.skip_base64_extensions = ('.nii', '.nii.gz', '.npy', '.h5')

from .breastgpt import *
from .template import *
import importlib
import os
import sys 
from lib.model.networks import AlexNet
from lib.utils import GetModel

# 현재 디렉토리 설정
current_dir = os.path.dirname(os.path.abspath(__file__))

# .faceswap 파일이 faker.py와 동일 폴더에 있다고 가정
faceswap_config_path = os.path.join(current_dir, ".faceswap")

current_path = '/content/drive/MyDrive/Faker/faceswap-master'
lib_path = os.path.join(current_path, 'lib')
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

# 디렉토리 경로 설정
directory = "C:\\Users\\User\\Desktop\\faker\\.fs_cache\\"

modules = [
    'lib.model.networks.simple_nets', 'lib.model.networks.clip',
    'lib.model.losses.feature_loss', 'lib.model.losses.loss',
    'lib.model.losses.perceptual_loss', 'lib.model.initializers',
    'lib.model.layers', 'lib.model.nn_blocks', 'lib.model.normalization',
    'lib.model.optimizers', 'lib.model.session', 'lib.model.autoclip',
    'lib.model.backup_restore'
]

for module in modules:
    importlib.import_module(module)

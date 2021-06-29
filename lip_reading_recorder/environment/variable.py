import os


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT_DIR,'environment','configuration.pickle')
DATASET_DIR = os.path.join(ROOT_DIR,'data','dataset')

MODEL_DIR = os.path.join(ROOT_DIR,'data','model')

MODEL_LIST = ['resnet','vggnet']

INPUT_SHAPE = {'resnet':(30,120,120,3)}



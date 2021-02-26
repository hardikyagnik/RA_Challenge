import os


DATA_DIR_PATH = os.path.join(os.path.dirname(__file__),'data')
BINARIES_PATH = os.path.join(os.path.dirname(__file__),'binaries')

TRAIN_DATA_PATH = os.path.join(DATA_DIR_PATH, 'Adult_Training_Set.csv')
TEST_DATA_PATH = os.path.join(DATA_DIR_PATH, 'Adult_Testing_Set.csv')

os.makedirs(BINARIES_PATH, exist_ok=True)

EPOCH = 2000
LR = 0.01
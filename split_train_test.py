import os
import shutil
from tqdm import tqdm

dirs = ['10s/', '20s/', '30s/', 'male/', 'female/']

for dir_ in tqdm(dirs):
    files = os.listdir(dir_)
    total_size = len(files)
    test_size = int(0.2*total_size)

    for i, file_ in enumerate(files):
        if i < test_size:
            shutil.copyfile(os.path.join(dir_, file_), os.path.join('test', file_))
        else:
            shutil.copyfile(os.path.join(dir_, file_), os.path.join('training', file_))

import shutil

import numpy as np
import os
import re
import torch
from torch.autograd import Variable

def read_list(list_file):
    f = open(list_file, "r")
    lines = f.readlines()
    list_sig = []
    for x in lines:
        list_sig.append(x[:-1])
    f.close()
    return list_sig
def gen_list(wav_dir, append):
    l = []
    lst = os.listdir(wav_dir)
    lst.sort()
    for f in lst:
        if re.search(append, f):
            l.append(f)
    return l

def write_log(file,name, train, validate):
    message = ''
    for m, val in enumerate(train):
        message += ' --TRerror%i=%.3f ' % (m, val.data.numpy())
    for m, val in enumerate(validate):
        message += ' --CVerror%i=%.3f ' % (m, val.data.numpy())
    file.write(name + ' ')
    file.write(message)
    file.write('/n')

def makedirs(dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

def saveConfig(yaml, yaml_name, src, dst):
    f_params = open(dst + '/' + yaml_name, 'w')
    for k, v in yaml.items():
        f_params.write('{}:\t{}\n'.format(k, v))
    shutil.copy(os.path.join(src,'train.py'), os.path.join(dst,'train.py'))
    shutil.copy(os.path.join(src,'test.py'), os.path.join(dst,'test.py'))


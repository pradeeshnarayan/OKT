import json
import torch
import os
import logging
import numpy as np

def save_config(obj, path):
    f = open(path, 'w')
    json.dump(obj.args, f, indent='  ')
    f.write('\n')
    f.close()


def load_config(Model, path):
    f = open(path, 'r')
    return Model(json.load(f))


def save_snapshot(model, ws, id,modelname):
    filename = os.path.join(ws, 'snapshots/'+modelname, 'model.%s' % str(id))
    f = open(filename, 'wb')
    torch.save(model.state_dict(), f)
    f.close()


def load_snapshot(model, ws, id,modelname):
    filename = os.path.join(ws, 'snapshots/'+modelname, 'model.%s' % str(id))
    f = open(filename, 'rb')
    model.load_state_dict(torch.load(f, map_location=lambda s, loc: s))
    f.close()


def load_last_snapshot(model, ws,modelname):
    last = 0
    path = os.path.join(ws, 'snapshots/'+modelname)
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    for file in os.listdir(path):
        if 'model.' in file:
            epoch = int(file.split('.')[1])
            if epoch > last:
                last = epoch
    if last > 0:
        load_snapshot(model, ws, last,modelname)
    return last


def open_result(ws, name, id):
    return open(os.path.join(ws, 'results', '%s.%s' %
                             (name, str(id))), 'w')

def load_embedding(filename):
    f = open(filename, encoding='utf-8')
    wcnt, emb_size = next(f).strip().split(' ')
    wcnt = int(wcnt)
    emb_size = int(emb_size)

    words = []
    embs = []
    for line in f:
        fields = line.strip().split(' ')
        word = fields[0]
        emb = np.array([float(x) for x in fields[1:]])
        words.append(word)
        embs.append(emb)

    embs = np.asarray(embs)
    return wcnt, emb_size, words, embs

use_cuda = torch.cuda.is_available()


def Variable(*args, **kwargs):
    v = torch.autograd.Variable(*args, **kwargs)
    if use_cuda:
        v = v.cuda()
    return v

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def colored(text, color, bold=False):
    if bold:
        return bcolors.BOLD + color + text + bcolors.ENDC
    else:
        return color + text + bcolors.ENDC


LOG_COLORS = {
    'WARNING': bcolors.WARNING,
    'INFO': bcolors.OKGREEN,
    'DEBUG': bcolors.OKBLUE,
    'CRITICAL': bcolors.WARNING,
    'ERROR': bcolors.FAIL
}


class ColoredFormatter(logging.Formatter):
    def __init__(self, msg, datefmt, use_color=True):
        logging.Formatter.__init__(self, msg, datefmt=datefmt)
        self.use_color = use_color

    def format(self, record):
        levelname = record.levelname
        if self.use_color and levelname in LOG_COLORS:
            record.levelname = colored(record.levelname[0],
                                       LOG_COLORS[record.levelname])
        return logging.Formatter.format(self, record)

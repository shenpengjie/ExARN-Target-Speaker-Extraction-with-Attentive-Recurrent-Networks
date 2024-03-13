import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import yaml
import shutil
import torch
import time
import argparse
import torch.nn as nn
import logging as log
import torch.multiprocessing

from pathlib import Path
from dataloader import BatchDataLoader, SpeechMixDataset
from utils.Checkpoint import Checkpoint
from networks.modify_arn_extractor import NET_Wrapper
from utils.progressbar import progressbar as pb
from utils.util import makedirs, saveConfig
from torch.cuda.amp import autocast, GradScaler
from criteria import sisdr_loss,sdr_loss,rimagcompressmse_sisdr


def validate(network, eval_loader, criterion):
    network.eval()
    with torch.no_grad():
        cnt = 0.
        accu_eval_loss = 0.0
        ebar = pb(0, len(eval_loader.get_dataloader()), 20)
        ebar.start()
        correct=0
        total=0
        for j, batch_eval in enumerate(eval_loader.get_dataloader()):
            features,anchors,mix_len,aux_len= batch_eval[0].cuda(), batch_eval[1].cuda(),batch_eval[3],batch_eval[4]
            # outputs=network(features,anchors,mix_len,aux_len)

            outputs= network(features,anchors,mix_len,aux_len)
            eval_loss=criterion(outputs,batch_eval).data.item()
            class_prob=outputs[1]
            idx=torch.argmax(class_prob,dim=-1)
            for k in range(batch_eval[0].size(0)):
                if idx[k]==batch_eval[5][k]:
                    correct+=1
                    total+=1
                else:
                    total+=1
                # for idx, cri in enumerate(criterion):
                # # loss += cri(outputs, class_prob,batch_eval) * weight[idx]
                #     loss += cri(outputs,  batch_eval)
            # eval_loss = loss.data.item()
            accu_eval_loss += eval_loss
            cnt += 1.
            ebar.update_progress(j, 'CV   ', 'loss:{:.5f}/{:.5f}'.format(eval_loss, accu_eval_loss / cnt))

        avg_eval_loss = accu_eval_loss / cnt
        print()
        print("正确率{:.2f}".format(correct/total*100),end='')
    print()

    network.train()
    return avg_eval_loss


if __name__ == '__main__':

    """
    environment part
    """
    # loading argument
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", help="trained model name, retrain if no input", default='none')
    parser.add_argument("-y", "--yaml_name", help="config file name")
    args = parser.parse_args()

    # loading config
    _abspath = Path(os.path.abspath(__file__)).parent
    _project = _abspath.stem
    _yaml_path = os.path.join(_abspath, 'configs/' + args.yaml_name)
    try:
        with open(_yaml_path, 'r') as f_yaml:
            config = yaml.load(f_yaml, Loader=yaml.FullLoader)
    except:
        raise ValueError('No config file found at "%s"' % _yaml_path)

    # make output dirs
    _outpath = config['OUTPUT_DIR'] + _project + config['WORKSPACE']
    _modeldir = _outpath + '/checkpoints/'
    _datadir = _outpath + '/estimations/'
    _logdir = _outpath + '/log/'
    makedirs([_modeldir, _datadir, _logdir])
    saveConfig(config, args.yaml_name, _abspath, _outpath)

    """
    network part
    """
    # dataset
    torch.multiprocessing.set_sharing_strategy('file_system')
    tr_mix_dataset = SpeechMixDataset(config, mode='train')
    tr_batch_dataloader = BatchDataLoader(tr_mix_dataset, config['BATCH_SIZE'], is_shuffle=True,
                                          workers_num=config['NUM_WORK'])
    if config['USE_CV']:
        cv_mix_dataset = SpeechMixDataset(config, mode='validate')
        cv_batch_dataloader = BatchDataLoader(cv_mix_dataset, config['BATCH_SIZE'], is_shuffle=False,
                                              workers_num=config['NUM_WORK'])

    # device setting
    os.environ["CUDA_VISIBLE_DEVICES"] = config['CUDA_ID']

    # set model and optimizer
    network = NET_Wrapper(config['FRAME_SIZE'], config['FRAME_SHIFT'], config['FEATURE_DIM'], config['HIDDEN_DIM'], infer=False, causal=False)
    network = nn.DataParallel(network)
    network.cuda()
    parameters = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print("Trainable parameters : " + str(parameters))
    optimizer = torch.optim.Adam(network.parameters(), lr=config['LR'],amsgrad=True)
    # scaler = GradScaler()
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        threshold=0.001,
        verbose=True,
    )
    # scaler = GradScaler()

    criterion = rimagcompressmse_sisdr()

    weight = [1.]

    if args.model_name == 'none':
        log.info('#' * 12 + 'NO EXIST MODEL, TRAIN NEW MODEL ' + '#' * 12)
        best_loss = float('inf')
        start_epoch = 0
    else:
        checkpoint = Checkpoint()
        checkpoint.load(args.model_name)
        start_epoch = checkpoint.start_epoch
        best_loss = checkpoint.best_loss
        network.load_state_dict(checkpoint.state_dict,strict=False)
        # optimizer.load_state_dict(checkpoint.optimizer)
        log.info('#' * 18 + 'Finish Resume Model ' + '#' * 18)

    """
    training part
    """
    log.info('#' * 20 + ' START TRAINING ' + '#' * 20)
    cnt = 0.  #
    for epoch in range(start_epoch, config['MAX_EPOCH']):

        # initial param
        accu_train_loss = 0.0
        network.train()
        tbar = pb(0, len(tr_batch_dataloader.get_dataloader()), 20)
        tbar.start()

        for i, batch_info in enumerate(tr_batch_dataloader.get_dataloader()):
            features, anchors, mix_len, aux_len = batch_info[0].cuda(), batch_info[1].cuda(), batch_info[3], batch_info[4]

            # forward + backward + optimize
            optimizer.zero_grad()
            # with autocast():
                # outputs,class_prob = network(features,anchors)
                # loss = criterion(outputs, class_prob, batch_info)
            # with autocast():

            outputs = network(features, anchors, mix_len, aux_len)
            loss = criterion(outputs, batch_info)
            loss.backward()
            # scaler.scale(loss).backward()
            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
            # scaler.step(optimizer)
            # scaler.update()
            optimizer.step()

            # calculate losses
            running_loss = loss.data.item()
            accu_train_loss += running_loss

            # display param
            cnt += 1
            del loss, outputs, batch_info

            tbar.update_progress(i, 'Train', 'epoch:{}/{}, loss:{:.5f}/{:.5f}'.format(epoch + 1,
                                                                                      config['MAX_EPOCH'], running_loss,
                                                                                      accu_train_loss / cnt))
            if config['USE_CV'] and (i + 1) % config['EVAL_STEP'] == 0:
                print()
                avg_train_loss = accu_train_loss / cnt
                avg_eval_loss = validate(network, cv_batch_dataloader, criterion)
                is_best = True if avg_eval_loss < best_loss else False
                best_loss = avg_eval_loss if is_best else best_loss
                log.info('Epoch [%d/%d], ( TrainLoss: %.4f | EvalLoss: %.4f )' % (
                    epoch + 1, config['MAX_EPOCH'], avg_train_loss, avg_eval_loss))
                lr_scheduler.step(avg_eval_loss)
                checkpoint = Checkpoint(epoch + 1, avg_train_loss, best_loss, network.state_dict(),
                                        optimizer.state_dict())
                model_name = _modeldir + '{}-{}-val.ckpt'.format(epoch + 1, i + 1)
                best_model = _modeldir + 'best.ckpt'
                if is_best:
                    checkpoint.save(is_best, best_model)
                if not config['SAVE_BEST_ONLY']:
                    checkpoint.save(False, model_name)
                accu_train_loss = 0.0
                network.train()
                cnt = 0.



    timeit = time.strftime('%Y-%m-%d-%H_', time.localtime(time.time()))
    log_path = str(_abspath) + '/train.log'
    if os.path.exists(log_path):
        shutil.copy(log_path, _outpath + '/log/' + timeit + 'train.log')
        file = open(log_path, 'w').close()

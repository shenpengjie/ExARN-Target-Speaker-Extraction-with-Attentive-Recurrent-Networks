import os
import yaml

import torch.nn as nn
import logging as log
from torch.cuda.amp import autocast, GradScaler
from datetime import timedelta
import torch
#ddp
import argparse
import torch.distributed as dist

from pathlib import Path
from dataloader import BatchDataLoader, SpeechMixDataset
from utils.Checkpoint import Checkpoint
from networks.modify_arn_extractor import NET_Wrapper
from utils.progressbar import progressbar as pb
from utils.util import makedirs, saveConfig
from collections import OrderedDict
from criteria import sisdr_loss,sdr_loss

from torch.cuda.amp import autocast,GradScaler
# def criterion(est,batch_info,rank):
#     raw = batch_info[1].cuda(rank)
#     mask_for_loss=batch_info[3].cuda(rank)
#     loss = torch.sum((est - raw) ** 2, dim=1) / torch.sum(mask_for_loss, dim=1)
#     return torch.mean(loss)
class ddp_train(object):
    def __init__(self,world_size,config,modeldir,model_name=None):
        self.world_size=world_size
        self.config=config
        self.modeldir=modeldir
        self.model_name=model_name
    def validate(self,network, eval_loader, weight, rank, *criterion):
        network.eval()
        with torch.no_grad():
            cnt = 0.
            accu_eval_loss = 0.0
            ebar = pb(0, len(eval_loader.get_dataloader()), 20)
            ebar.start()
            for j, batch_eval in enumerate(eval_loader.get_dataloader()):
                features, anchors, mix_len, aux_len = batch_eval[0].cuda(), batch_eval[1].cuda(), batch_eval[3], \
                                                      batch_eval[4]
                outputs = network(features,anchors,mix_len,aux_len)
                loss = 0.
                for idx, cri in enumerate(criterion):
                    loss += cri(outputs, batch_eval,rank) * weight[idx]
                eval_loss = loss.data.item()
                accu_eval_loss += eval_loss
                cnt += 1.
                ebar.update_progress(j, 'CV   ', 'loss:{:.5f}/{:.5f}'.format(eval_loss, accu_eval_loss / cnt))

            avg_eval_loss = accu_eval_loss / cnt
        print()
        network.train()
        return avg_eval_loss

    def start_train(self, rank=0):
        # dataset
        tr_mix_dataset = SpeechMixDataset(self.config, mode='train')
        # tr_mix_dataset_dist=torch.utils.data.distributed.DistributedSampler(tr_mix_dataset,rank=2)

        if self.world_size>1:
            # datasampler=None
            datasampler = torch.utils.data.distributed.DistributedSampler(tr_mix_dataset)
            tr_batch_dataloader = BatchDataLoader(tr_mix_dataset, self.config['BATCH_SIZE'], is_shuffle=(datasampler is None),
                                              workers_num=self.config['NUM_WORK'], sampler=datasampler)
        else:
            tr_batch_dataloader=BatchDataLoader(tr_mix_dataset,self.config['BATCH_SIZE'],is_shuffle=True,workers_num=self.config['num_work'])


        if self.config['USE_CV'] and rank==0:
            cv_mix_dataset = SpeechMixDataset(self.config, mode='validate')
            cv_batch_dataloader = BatchDataLoader(cv_mix_dataset, self.config['BATCH_SIZE'], is_shuffle=False,
                                                  workers_num=self.config['NUM_WORK'])

        # set model and optimizer
        network = NET_Wrapper(self.config['FRAME_SIZE'], self.config['FRAME_SHIFT'], self.config['FEATURE_DIM'], self.config['HIDDEN_DIM'],
                              infer=False, causal=False)

        cur_device = rank
        network.to(cur_device)
        optimizer = torch.optim.Adam(network.parameters(), lr=self.config['LR'], amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.983, last_epoch=-1)
        scaler = GradScaler()
        criterion = sisdr_loss()
        # criterion1 =sisdr_loss()
        weight = [1.]


        network = nn.parallel.DistributedDataParallel(network, device_ids=[cur_device], output_device=0)
        if self.model_name == 'none':
            if rank == 0:
                log.info('#' * 12 + 'NO EXIST MODEL, TRAIN NEW MODEL ' + '#' * 12)

            best_loss = float('inf')
            start_epoch = 0
        else:
            checkpoint = Checkpoint()
            checkpoint.load(self.model_name)
            start_epoch = checkpoint.start_epoch
            best_loss = checkpoint.best_loss
            # new_state=OrderedDict([(key[7:],value) for key,value in checkpoint.state_dict.items()])
            network.load_state_dict(checkpoint.state_dict)
            optimizer.load_state_dict(checkpoint.optimizer)
            if rank == 0:
                log.info('#' * 18 + 'Finish Resume Model ' + '#' * 18)

        if rank == 0:
            parameters = sum(p.numel() for p in network.parameters() if p.requires_grad)
            print("Trainable parameters : " + str(parameters))




        """
        training part
        """
        # best_loss = float('inf')
        # start_epoch = 0
        if rank == 0:
            log.info('#' * 20 + ' START TRAINING ' + '#' * 20)

        cnt = 0.  #

        for epoch in range(start_epoch, self.config['MAX_EPOCH']):
            # set learning rate for every epoch
            # initial param
            accu_train_loss = 0.0
            network.train()
            tbar = pb(0, len(tr_batch_dataloader.get_dataloader()), 20)
            tbar.start()

            for i, batch_info in enumerate(tr_batch_dataloader.get_dataloader()):
                features, anchors, mix_len, aux_len = batch_info[0].cuda(rank), batch_info[1].cuda(rank), batch_info[3], batch_info[4]
                # forward + backward + optimize
                optimizer.zero_grad()
                # with autograd.detect_anomaly():
                with autocast():
                    outputs = network(features, anchors, mix_len, aux_len)
                    loss = criterion(outputs, batch_info,rank)
                # loss=c
                    #
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
                scaler.step(optimizer)
                scaler.update()
                # loss.backward()

                # optimizer.step()

                dist.all_reduce(loss)
                # calculate losses
                running_loss = loss.data.item()
                accu_train_loss += running_loss
                # display param
                cnt += 1
                del loss, outputs, batch_info
                if rank == 0:
                    tbar.update_progress(i, 'Train', 'epoch:{}/{}, loss:{:.5f}/{:.5f}'.format(epoch + 1,
                                                                                              self.config['MAX_EPOCH'],
                                                                                              running_loss,
                                                                                              accu_train_loss / cnt))

                if self.config['USE_CV'] and (i + 1) % self.config['EVAL_STEP'] == 0:
                    if epoch > 33:
                        scheduler.step()
                    if rank==0:
                        print()
                        avg_train_loss = accu_train_loss / cnt
                        avg_eval_loss = self.validate(network, cv_batch_dataloader, weight, rank, criterion)
                        is_best = True if avg_eval_loss < best_loss else False
                        best_loss = avg_eval_loss if is_best else best_loss

                        log.info('Epoch [%d/%d], ( TrainLoss: %.4f | EvalLoss: %.4f )' % (
                            epoch + 1, self.config['MAX_EPOCH'], avg_train_loss, avg_eval_loss))

                        checkpoint = Checkpoint(epoch + 1, avg_train_loss, best_loss, network.state_dict(),
                                                        optimizer.state_dict())
                        model_name = self.modeldir + '{}-{}-val.ckpt'.format(epoch + 1, i + 1)
                        best_model = self.modeldir + 'best.ckpt'
                        if is_best:
                            checkpoint.save(is_best, best_model)
                        if not self.config['SAVE_BEST_ONLY']:
                            checkpoint.save(False, model_name)


                    accu_train_loss = 0.0
                    network.train()
                    cnt = 0.


    def lanch_prcess(self,rank,func):
        dist.init_process_group(backend=dist.Backend.NCCL, world_size=self.world_size, rank=rank, timeout=timedelta(seconds=5))
        torch.manual_seed(12345)
        func(rank)
        self.cleanup()

    def cleanup(self):
        dist.destroy_process_group()

    def lanch_job(self):
        if self.world_size > 1:
            torch.multiprocessing.spawn(
                self.lanch_prcess,
                nprocs=self.world_size,
                args=(
                    self.start_train,
                ),
            )
        else:
            self.start_train()

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


    # device setting
    os.environ["CUDA_VISIBLE_DEVICES"] = config['CUDA_ID']
    os.environ["MASTER_ADDR"]='127.0.0.1'
    os.environ["MASTER_PORT"]='2959373'
    # launch_job(num_proc=2,config=config,yaml=args.yaml_name)

    # make output dirs
    _abspath = Path(os.path.abspath(__file__)).parent
    _project = _abspath.stem
    _outpath = config['OUTPUT_DIR'] + _project + config['WORKSPACE']
    _modeldir = _outpath + '/checkpoints/'
    _datadir = _outpath + '/estimations/'
    _logdir = _outpath + '/log/'
    makedirs([_modeldir, _datadir, _logdir])
    saveConfig(config, args.yaml_name, _abspath, _outpath)
    """
    network part
    """
    ddp=ddp_train(world_size=2,config=config,modeldir=_modeldir,model_name=args.model_name)
    ddp.lanch_job()






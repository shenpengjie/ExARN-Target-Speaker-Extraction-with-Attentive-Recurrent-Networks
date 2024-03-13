import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import yaml
import soundfile as sf
import logging as log
import numpy as np

from pathlib import Path
from torch import nn
# from metrics import Metrics
from networks.modify_arn_extractor import NET_Wrapper
from utils.Checkpoint import Checkpoint
from utils.progressbar import progressbar as pb
from utils.stft_istft import STFT
from utils.util import makedirs, gen_list
from torch.autograd.variable import *
from utils.getPESQ import getPESQ
import mir_eval
import torch.multiprocessing as mp


class Test(object):
    def __init__(self, inpath, outpath, type='online', suffix='mix.wav',EPSILON=0.0000001):
        self.inpath = inpath
        self.outpath = outpath
        self.type = type
        self.suffix = suffix
        self.EPSILON=EPSILON
        self.too_path="/home/imu_speech1/spj/code/tools/"
        self.pool=mp.Pool(20)

    def forward(self, network):
        network.eval()
        tt_lst = gen_list(self.inpath+'/mix', self.suffix)
        tt_len = len(tt_lst)
        pbar = pb(0, tt_len)
        pbar.start()

        for i in range(tt_len):
            pbar.update_progress(i, 'tt', '')
            self.pool.apply_async(func=self.process_wav(tt_lst[i]),args=(self,))

        pbar.finish()
    def process_wav(self,wav_name):
        mix, fs = sf.read(self.inpath + 'mix/' + wav_name)
        aux, fs = sf.read(self.inpath + 'aux/' + wav_name)
        raw, fs = sf.read(self.inpath + 's1/' + wav_name)
        aux_alpha_pow = 1 / (np.sqrt(np.sum(aux ** 2) / len(aux)) + self.EPSILON)
        alpha_pow = 1 / (np.sqrt(np.sum(mix ** 2) / len(mix)) + self.EPSILON)
        mix = mix * alpha_pow
        aux = aux * aux_alpha_pow
        mixture = Variable(torch.FloatTensor(mix.astype('float32')))
        anchor = Variable(torch.FloatTensor(aux.astype('float32')))
        mixture = mixture.reshape([1, -1])
        anchor = anchor.reshape([1, -1])
        mix_len = torch.tensor([mixture.size(1) // 16], dtype=torch.long)
        aux_len = torch.tensor([anchor.size(1) // 16], dtype=torch.long)


        """------------------------------------modify  area------------------------------------"""
        with torch.no_grad():
            est = network(mixture, anchor, mix_len, aux_len)
        est_t = est[0][0].data.cpu().squeeze().numpy()
        """------------------------------------modify  area------------------------------------"""
        if self.type == 'online':
            sf.write(self.outpath + wav_name[:-len(self.suffix)] + '_clean.wav', raw[:est_t.size], fs)
        sf.write(self.outpath + wav_name[:-len(self.suffix)] + '_est.wav', est_t / alpha_pow, fs)
        sf.write(self.outpath + wav_name[:-len(self.suffix)] + '_mix.wav', mix[:est_t.size] / alpha_pow, fs)


    def compute_pesq(self):
        mix_lst = gen_list(self.outpath, '_mix.wav')
        estpesq_list = []
        mixpesq_list = []
        # mix_lst.remove('446c0209_1.8488_22hc010z_-1.8488_446o0305_mix.wav')
        mix_len = len(mix_lst)
        for i in range(mix_len):
            # if i==2687:
            #     continue
            mix_path=self.outpath + mix_lst[i][:-8] + '_mix.wav'
            est_path=self.outpath + mix_lst[i][:-8] + '_est.wav'
            clean_path=self.outpath + mix_lst[i][:-8] + '_clean.wav'
            est_pesq=getPESQ(self.too_path,clean_path,est_path)
            mix_pesq=getPESQ(self.too_path,clean_path,mix_path)
            estpesq_list.append(est_pesq)
            mixpesq_list.append(mix_pesq)

        mixpesq_list = np.array(mixpesq_list)
        estpesq_list = np.array(estpesq_list)
        mixpesq_mean = np.mean(mixpesq_list)
        estpesq_mean = np.mean(estpesq_list)
        print('\nmixpesq is ' + str(mixpesq_mean) + '\nestpesq is ' + str(estpesq_mean))

    def pow_np_norm(self,inp):
        return np.square(np.linalg.norm(inp, ord=2))

    def pow_norm(self,s1, s2):
        return np.sum(s1 * s2)

    def si_snr(self,est, ref, eps=1e-8):

        s_ref = (self.pow_norm(est, ref) / (self.pow_np_norm(ref) + eps)) * ref
        e_nse = est - s_ref
        return 10 * np.log10((self.pow_np_norm(s_ref) / (self.pow_np_norm(e_nse) + eps)) + eps)

    def compute_sdr_sisdr(self):
        mix_lst = gen_list(self.outpath, '_mix.wav')
        # mix_lst.remove('446c0209_1.8488_22hc010z_-1.8488_446o0305_mix.wav')
        mix_len = len(mix_lst)
        estsdr_list = []
        mixsdr_list = []
        estsisnr_list=[]
        mixsisnr_list=[]
        # error_list=[]
        # bad_list=[]
        for i in range(mix_len):
            mix, fs = sf.read(self.outpath + mix_lst[i][:-8] + '_mix.wav')
            clean, fs = sf.read(self.outpath + mix_lst[i][:-8] + '_clean.wav')
            est, fs = sf.read(self.outpath + mix_lst[i][:-8] + '_est.wav')
            est = est - np.mean(est, axis=-1, keepdims=True)
            clean = clean - np.mean(clean, axis=-1, keepdims=True)
            mix = mix -np.mean(mix, axis=-1,keepdims=True)
            est_sdr = 10 * np.log10(np.sum(clean ** 2) / (np.sum((clean - est) ** 2) + self.EPSILON) + self.EPSILON)
            mix_sdr = 10 * np.log10(np.sum(clean ** 2) / (np.sum((clean - mix) ** 2) + self.EPSILON) + self.EPSILON)
            # est_sdr = mir_eval.separation.bss_eval_sources(clean,est)
            # mix_sdr = mir_eval.separation.bss_eval_sources(clean,mix)
            est_sisnr=self.si_snr(est,clean)
            mix_sisnr=self.si_snr(mix,clean)
            mixsdr_list.append(mix_sdr)
            estsdr_list.append(est_sdr)
            estsisnr_list.append(est_sisnr)
            mixsisnr_list.append(mix_sisnr)
            # if est_sisnr<0:
            #     error_list.append({"name":mix_lst[i],"est_sisdr": est_sisnr})
            #     # sf.write(mix_lst[i][:-8]+'_mix.wav',mix,fs)
            #     # sf.write(mix_lst[i][:-8] + '_clean.wav', clean, fs)
            #     # sf.write(mix_lst[i][:-8] + '_est.wav', est, fs)
            # elif est_sisnr<12:
            #     bad_list.append({"name":mix_lst[i],"est_sisdr": est_sisnr})
        mixsdr_list = np.array(mixsdr_list)
        estsdr_list = np.array(estsdr_list)
        estsisnr_list=np.array(estsisnr_list)
        mixsisnr_list=np.array(mixsisnr_list)
        mixsdr_mean = np.mean(mixsdr_list)
        estsdr_mean = np.mean(estsdr_list)
        mixsisnr_mean=np.mean(mixsisnr_list)
        estsisnr_mean=np.mean(estsisnr_list)
        sisdri_list = estsisnr_list - mixsisnr_list
        count_less_than_zero = 0
        count_0_to_5 = 0
        count_5_to_10 = 0
        count_greater_than_10 = 0
        for value in sisdri_list:
            if value < 0:
                count_less_than_zero += 1
            elif 0 <= value < 5:
                count_0_to_5 += 1
            elif 5 <= value < 10:
                count_5_to_10 += 1
            else:
                count_greater_than_10 += 1
        print("小于0的数量：", count_less_than_zero)
        print("大于等于0小于5的数量：", count_0_to_5)
        print("大于等于5小于10的数量：", count_5_to_10)
        print("大于等于10的数量：", count_greater_than_10)
        print('\nmixsdr is ' + str(mixsdr_mean) + '\nestsdr is ' + str(estsdr_mean))
        print('\nmixsisdr is ' + str(mixsisnr_mean) + '\nestsisdr is ' + str(estsisnr_mean))
        # return error_list




if __name__ == '__main__':
    """
        environment part
        """
    # loading argument
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", help="trained model name, retrain if no input", default='none')
    parser.add_argument("-y", "--yaml_name", help="config file name")
    parser.add_argument("-n", "--new_test",
                        help="generate new test date[1] or only compute metrics with exist data[0]", default=0,
                        type=int)
    args = parser.parse_args()

    # loading config
    _abspath = Path(os.path.abspath(__file__)).parent
    _project = _abspath.stem
    _yaml_path = os.path.join(_abspath, 'configs/' + args.yaml_name)
    with open(_yaml_path, 'r') as f_yaml:
        config = yaml.load(f_yaml, Loader=yaml.FullLoader)

    # if online test
    _outpath = config['OUTPUT_DIR'] + _project + config['WORKSPACE']
    outpath = _outpath + '/estimations/'
    makedirs([outpath])
    EPSILON=config['EPSILON']
    os.environ["CUDA_VISIBLE_DEVICES"] = config['TT_CUDA']
    inpath = config['TEST_PATH']
    outpath = "/data02/exp_result/SE_out/"
    test = Test(inpath=inpath, outpath=outpath, type='online', suffix='.wav', EPSILON=EPSILON)
    if args.new_test:
        network = NET_Wrapper(config['FRAME_SIZE'], config['FRAME_SHIFT'], config['FEATURE_DIM'],  config['HIDDEN_DIM'], infer=False, causal=False)
        network = nn.DataParallel(network)
        network.cuda()
        checkpoint = Checkpoint()
        checkpoint.load(args.model_name)
        network.load_state_dict(checkpoint.state_dict)
        log.info('#' * 14 + 'Finish Resume Model For Test' + '#' * 14)
        print(checkpoint.best_loss)
        test.forward(network)
    test.compute_sdr_sisdr()
    test.compute_pesq()
    # print('error_num:' + str(len(error_list)))
    # print(error_list)
    # # cal metrics


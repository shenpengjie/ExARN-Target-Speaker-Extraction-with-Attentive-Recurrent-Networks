import random

import soundfile as sf
import numpy as np
import torch
import struct
import mmap
import librosa

from torch.nn.utils.rnn import *
from torch.autograd.variable import *
from torch.utils.data import Dataset, DataLoader
from utils.util import gen_list


class SpeechMixDataset(Dataset):
    def __init__(self, config, mode='train'):
        self.config = config
        self.wav_lst = config['TR_SPEECH_PATH'] if mode == 'train' else config['CV_SPEECH_PATH']
        self.frame_shift = config['FRAME_SHIFT']
        self.mix_speech_lst=gen_list(self.wav_lst+"mix",".wav")
        self.anchor_speech_lst=gen_list(self.wav_lst+"aux",".wav")
        self.target_speech_lst=gen_list(self.wav_lst+"s1",".wav")
        self.wav_len = len(self.mix_speech_lst)
        self.speech_len = config['SPEECH_LEN']
        self.speaker_label=dict()
        i=0
        self.speaker_lst=config['SPEECH_LST']
        with open(self.speaker_lst,'r',encoding='utf-8') as fin:
            for line in fin:
                self.speaker_label[line.strip()]=i
                i+=1

    def __len__(self):
        return self.wav_len

    def __getitem__(self, idx):
        mix_wav_name = self.mix_speech_lst[idx].strip()
        mix, fs = sf.read(self.wav_lst+"mix/"+mix_wav_name)
        anchor_wav_name=self.anchor_speech_lst[idx].strip()
        anchor,fs=sf.read(self.wav_lst+"aux/"+anchor_wav_name)
        s1_wav_name=self.target_speech_lst[idx].strip()
        s1,fs=sf.read(self.wav_lst+"s1/"+s1_wav_name)
        label=self.speaker_label[anchor_wav_name.split("_")[0][:3]]

        if len(mix)>self.speech_len:
            start=np.random.randint(0,len(mix)-self.speech_len)
            mix=mix[start:start+self.speech_len]
            s1=s1[start:start+self.speech_len]
            mix_len=(len(s1)//self.frame_shift)
        else:
            mix_len = (len(s1)//self.frame_shift)
            mix=np.concatenate((mix,np.zeros(self.speech_len-len(mix))))
            s1=np.concatenate((s1,np.zeros(self.speech_len-len(s1))))




        aux_alpha_pow=1 / (np.sqrt(np.sum(anchor ** 2) / len(anchor)) + self.config['EPSILON'])
        alpha_pow=1 / (np.sqrt(np.sum(mix ** 2) / len(mix)) + self.config['EPSILON'])
        anchor=anchor*aux_alpha_pow
        s1=s1*alpha_pow
        mix=mix*alpha_pow
        aux_len=(len(anchor)//self.frame_shift)

        sample = (Variable(torch.FloatTensor(mix.astype('float32'))),
                  Variable(torch.FloatTensor(anchor.astype('float32'))),
                  Variable(torch.FloatTensor(s1.astype('float32'))),
                  mix_len,
                  aux_len,
                  label
                  )

        return sample

class BatchDataLoader(object):
    def __init__(self, s_mix_dataset, batch_size, is_shuffle=True, workers_num=16,sampler=None):
        self.dataloader = DataLoader(s_mix_dataset, batch_size=batch_size, shuffle=is_shuffle, num_workers=workers_num,
                                     collate_fn=self.collate_fn,sampler=sampler)

    def get_dataloader(self):
        return self.dataloader

    @staticmethod
    def collate_fn(batch):
        batch.sort(key=lambda x: x[3], reverse=True)
        mix, aux, ref,mix_len,aux_len,label = zip(*batch)
        label=torch.tensor(list(label),dtype=torch.long)
        mix_len=torch.tensor(list(mix_len))
        aux_len=torch.tensor(list(aux_len))
        mix = pad_sequence(mix, batch_first=True)
        aux = pad_sequence(aux, batch_first=True)
        ref = pad_sequence(ref, batch_first=True)
        # noise = pad_sequence(noise, batch_first=True)
        # mixture = speech + noise
        return [mix, aux, ref, mix_len, aux_len,label]

class SpeechMixDataset_fix(Dataset):
    def __init__(self, config, mode='train'):
        self.config = config
        self.mode = mode
        self.frame_shift = config['FRAME_SHIFT']
        self.speaker_lst = config['SPEECH_LST']
        self.speaker_label = dict()
        i=0
        with open(self.speaker_lst, 'r', encoding='utf-8') as fin:
            for line in fin:
                self.speaker_label[line.strip()] = i
                i += 1
        if self.mode == 'train':
            self.wav_len=20000
            self.speech_file=config['TR_SPEECH_BIN']
            self.mm_list=[]
            for wav_id in list(self.speaker_label.keys()):
                f=open(self.speech_file+"/"+wav_id+".bin", 'r+b')
                mm = mmap.mmap(f.fileno(), 0)
                self.mm_list.append(mm)

        else:
            self.wav_lst = config['TR_SPEECH_PATH'] if mode == 'train' else config['CV_SPEECH_PATH']
            self.mix_speech_lst=gen_list(self.wav_lst+"mix",".wav")
            self.anchor_speech_lst=gen_list(self.wav_lst+"aux",".wav")
            self.target_speech_lst=gen_list(self.wav_lst+"s1",".wav")
            self.wav_len = len(self.mix_speech_lst)
        self.speech_len = config['SPEECH_LEN']



    def __len__(self):
        return self.wav_len

    def __getitem__(self, idx):
        if self.mode=='train':
            target_id=random.randint(0,100)
            other_id=random.randint(0,100)
            while other_id==target_id:
                other_id = random.randint(0, 100)
            snr=random.uniform(-5,5)
            weight_1 = 10**(snr/20)
            weight_2 = 10**(-snr / 20)
            label = target_id
            anchor_len = random.randint(20000,100000)
            mm_s1 = self.mm_list[target_id]
            len_s1 = int(len(mm_s1) / 4)
            s1_start=random.randint(0,len_s1-self.speech_len-1)
            s1=np.array(
            list(struct.unpack('f' * self.speech_len, mm_s1[s1_start * 4:s1_start * 4 + self.speech_len * 4])))
            # s1=librosa.resample(s1,orig_sr=16000,target_sr=8000,fix=True,scale=False)
            if s1_start<anchor_len:
                range1=(0,0)
            else:
                range1=range(0,s1_start-anchor_len)
            if (len_s1-anchor_len)<(s1_start+self.speech_len):
                range2=(0,0)
            else:
                range2 = range(s1_start+self.speech_len,len_s1-anchor_len)
            anchor_start=random.choice(list(range1)+list(range2))

            anchor=np.array(
            list(struct.unpack('f' * anchor_len, mm_s1[anchor_start * 4:anchor_start * 4 + anchor_len * 4])))
            # anchor = librosa.resample(anchor, orig_sr=16000, target_sr=8000,fix=True,scale=False)

            mm_s2 = self.mm_list[other_id]
            len_s2 = int(len(mm_s2) / 4)
            s2_start=random.randint(0,len_s2-self.speech_len-1)
            s2 =np.array(
            list(struct.unpack('f' * self.speech_len, mm_s2[s2_start * 4:s2_start * 4 + self.speech_len * 4])))
            # s2 = librosa.resample(s2, orig_sr=16000, target_sr=8000,fix=True,scale=False)

            s1 = weight_1 * s1;
            s2 = weight_2 * s2;

            mix=s1+s2
            max_amp_8k = max(max(s1),max(s2),max(mix),max(anchor));
            mix_scaling_8k = 1 / max_amp_8k * 0.9;
            s1 =s1 *mix_scaling_8k
            mix = mix*mix_scaling_8k
            anchor=anchor*mix_scaling_8k
            mix_len=(len(s1)//self.frame_shift)


        else:
            mix_wav_name = self.mix_speech_lst[idx].strip()
            mix, fs = sf.read(self.wav_lst+"mix/"+mix_wav_name)
            anchor_wav_name=self.anchor_speech_lst[idx].strip()
            anchor,fs=sf.read(self.wav_lst+"aux/"+anchor_wav_name)
            s1_wav_name=self.target_speech_lst[idx].strip()
            s1,fs=sf.read(self.wav_lst+"s1/"+s1_wav_name)
            label=self.speaker_label[anchor_wav_name.split("_")[0][:3]]

            if len(mix)>self.speech_len:
                start=np.random.randint(0,len(mix)-self.speech_len)
                mix=mix[start:start+self.speech_len]
                s1=s1[start:start+self.speech_len]
                mix_len=(len(s1)//self.frame_shift)
            else:
                mix_len = (len(s1)//self.frame_shift)
                mix=np.concatenate((mix,np.zeros(self.speech_len-len(mix))))
                s1=np.concatenate((s1,np.zeros(self.speech_len-len(s1))))




        aux_alpha_pow=1 / (np.sqrt(np.sum(anchor ** 2) / len(anchor)) + self.config['EPSILON'])
        alpha_pow=1 / (np.sqrt(np.sum(mix ** 2) / len(mix)) + self.config['EPSILON'])
        anchor=anchor*aux_alpha_pow
        s1=s1*alpha_pow
        mix=mix*alpha_pow
        aux_len=(len(anchor)//self.frame_shift)

        sample = (Variable(torch.FloatTensor(mix.astype('float32'))),
                  Variable(torch.FloatTensor(anchor.astype('float32'))),
                  Variable(torch.FloatTensor(s1.astype('float32'))),
                  mix_len,
                  aux_len,
                  label
                  )

        return sample



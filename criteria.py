import torch
import math
from utils.stft_istft_real_imag_hamming import STFT as complex_STFT
from utils.stft_istft import STFT as mag_STFT
from torch.autograd import Variable

class stftm_loss(object):
    def __init__(self, frame_size=320, frame_shift=160, loss_type='mae'):
        super(stftm_loss, self).__init__()
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.loss_type = loss_type
        self.complex_stft = complex_STFT(self.frame_size, self.frame_shift).cuda()

    def __call__(self, est, data_info):
        # est : est waveform
        mask = data_info[3].cuda()
        est_spec = self.complex_stft.transform(est)
        raw_spec = self.complex_stft.transform(data_info[1].cuda())

        est_spec = torch.abs(est_spec[..., 0]) + torch.abs(est_spec[..., 1])
        raw_spec = torch.abs(raw_spec[..., 0]) + torch.abs(raw_spec[..., 1])

        if self.loss_type == 'mse':
            loss = torch.sum((est_spec - raw_spec) ** 2) / torch.sum(mask)
        elif self.loss_type == 'mae':
            loss = torch.sum(torch.abs(est_spec - raw_spec)) / torch.sum(mask)

        return loss

class mag_loss(object):
    def __init__(self, frame_size=320, frame_shift=160, loss_type='mae'):
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.loss_type = loss_type
        self.mag_stft = mag_STFT(self.frame_size, self.frame_shift)

    def __call__(self, est, data_info):
        # est : [est_mag,noisy_phase]
        # data_info : [mixture,speech,noise,mask,nframe,len_speech]
        mask = data_info[3].cuda()

        raw_mag = self.mag_stft.transform(data_info[1])[0].permute(0, 2, 1).cuda()
        est_mag = est[0]
        if self.loss_type == 'mse':
            loss = torch.sum((est_mag - raw_mag) ** 2) / torch.sum(mask)
        elif self.loss_type == 'mae':
            loss = torch.sum(torch.abs(est_mag - raw_mag)) / torch.sum(mask)
        return loss

class wavemse_loss(object):
    def __call__(self, est, data_info):
        # mask = data_info[3].cuda()
        raw = data_info[1].cuda()
        loss = torch.sum((est - raw) ** 2,dim=1) / torch.sum(torch.ones_like(est),dim=1)
        return torch.mean(loss)

class pcm_loss(object):
    def __init__(self, frame_size=1024, frame_shift=512):
        super(pcm_loss, self).__init__()
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.complex_stft = complex_STFT(self.frame_size, self.frame_shift).cuda()

    def __call__(self, est, data_info):
        # est : est waveform
        mask = data_info[3].cuda()
        est_spec = self.complex_stft.transform(est)
        raw_spec = self.complex_stft.transform(data_info[1].cuda())

        noise_spec=self.complex_stft.transform(data_info[2].cuda())
        est_noise_spec=self.complex_stft.transform(data_info[0].cuda()-est)
        est_spec = torch.abs(est_spec[..., 0]) + torch.abs(est_spec[..., 1])
        raw_spec = torch.abs(raw_spec[..., 0]) + torch.abs(raw_spec[..., 1])
        est_noise_spec = torch.abs(est_noise_spec[..., 0]) + torch.abs(est_noise_spec[..., 1])
        noise_spec = torch.abs(noise_spec[..., 0]) + torch.abs(noise_spec[..., 1])
        loss = 0.5 * (torch.sum(torch.abs(est_spec - raw_spec)) / torch.sum(mask)) + 0.5 * (
                torch.sum(torch.abs(est_noise_spec - noise_spec)) / torch.sum(mask))
        return loss

class sdr_loss(object):
    def __init__(self):
        self.EPSILON = 1e-7
        self.crossentropy=torch.nn.CrossEntropyLoss()

    def __call__(self, est, data_info):
        ref = data_info[2].cuda()
        batch,_=ref.shape
        mask=data_info[3]
        ref_ori=[]
        est_ori=[]
        for i in range(batch):
            ref_ori.append(ref[i,:mask[i]])
            est_ori.append(est[i,:mask[i]])
        val_lst=[]
        for i in range(batch):
            noise = ref_ori[i] - est_ori[i]
            val = 10 * torch.log10((torch.sum(ref_ori[i] ** 2) ) / (torch.sum(noise ** 2) + 1.0e-6)+ 1.0e-6)
            val_lst.append(val)

        # return - torch.mean(torch.stack(val_lst))+0.5*self.crossentropy(prob,data_info[4].cuda())
        return - torch.mean(torch.stack(val_lst))

class sisdr_loss(object):
    def __init__(self):
        self.EPSILON = 1e-7
        self.crossentropy=torch.nn.CrossEntropyLoss()
        self.frame_shift=16
    def pow_np_norm(self,inp):
        return torch.square(torch.linalg.norm(inp))

    def pow_norm(self,s1, s2):
        return torch.sum(s1 * s2)
    def __call__(self, batch_est ,data_info):
        batch_ref=data_info[2].cuda()
        mix_len=data_info[3]
        batch,_=batch_est.shape
        sisdr_lst=[]
        for i in range(batch):
            est=batch_est[i,:mix_len[i]*16]
            ref=batch_ref[i,:mix_len[i]*16]
            s_ref = (self.pow_norm(est, ref) / (self.pow_np_norm(ref) + self.EPSILON)) * ref
            e_nse = est - s_ref
            sisdr_lst.append(10 * torch.log10((self.pow_np_norm(s_ref) / (self.pow_np_norm(e_nse) + self.EPSILON)) + self.EPSILON))
        return -torch.mean(torch.stack(sisdr_lst))

class rimagcompressmse_sisdr(object):
    def __init__(self, frame_size=256, frame_shift=128, loss_type='mae',m=0.2,s=30):
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.loss_type = loss_type
        self.complex_stft = complex_STFT(self.frame_size, self.frame_shift).cuda()
        self.eps = 1e-8
        self.m=m
        self.s=s
        self.ce=torch.nn.CrossEntropyLoss()
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m
    def pow_np_norm(self, inp):
        return torch.square(torch.linalg.norm(inp))
    def pow_norm(self, s1, s2):
        return torch.sum(s1 * s2)
    def RI_Mag_Compress_Mse(self, est_real, est_imag, ref_real, ref_imag, lamda=0.5):
        '''
            est_real: N x F x T
        '''

        est_mag = torch.sqrt(torch.clamp(est_real ** 2 + est_imag ** 2, self.eps))
        ref_mag = torch.sqrt(torch.clamp(ref_real ** 2 + ref_imag ** 2, self.eps))

        press_est_mag = torch.pow(est_mag, lamda)
        press_ref_mag = torch.pow(ref_mag, lamda)

        est_pha = torch.atan2(est_imag + self.eps, est_real + self.eps)
        ref_pha = torch.atan2(ref_imag + self.eps, ref_real + self.eps)

        press_est_real = press_est_mag * torch.cos(est_pha)
        press_est_imag = press_est_mag * torch.sin(est_pha)
        press_ref_real = press_ref_mag * torch.cos(ref_pha)
        press_ref_imag = press_ref_mag * torch.sin(ref_pha)

        mag_loss = torch.square(press_est_mag - press_ref_mag)
        real_loss = torch.square(press_est_real - press_ref_real)
        imag_loss = torch.square(press_est_imag - press_ref_imag)
        loss = mag_loss + real_loss + imag_loss


        loss = torch.mean(loss)
        return 10*loss

    def RI_Mag_Compress_Mse_ASYM(self, est_real, est_imag, ref_real, ref_imag, lamda=0.5):
        est_mag = torch.sqrt(torch.clamp(est_real ** 2 + est_imag ** 2, self.eps))
        ref_mag = torch.sqrt(torch.clamp(ref_real ** 2 + ref_imag ** 2, self.eps))

        press_est_mag = torch.pow(est_mag, lamda)
        press_ref_mag = torch.pow(ref_mag, lamda)

        est_pha = torch.atan2(est_imag + self.eps, est_real + self.eps)
        ref_pha = torch.atan2(ref_imag + self.eps, ref_real + self.eps)

        press_est_real = press_est_mag * torch.cos(est_pha)
        press_est_imag = press_est_mag * torch.sin(est_pha)
        press_ref_real = press_ref_mag * torch.cos(ref_pha)
        press_ref_imag = press_ref_mag * torch.sin(ref_pha)

        mag_loss = torch.square(torch.max(press_est_mag - press_ref_mag, Variable(torch.zeros_like(press_est_mag))))
        real_loss = torch.square(torch.max(press_est_real - press_ref_real, Variable(torch.zeros_like(press_ref_real))))
        imag_loss = torch.square(torch.max(press_est_imag - press_ref_imag, Variable(torch.zeros_like(press_ref_imag))))
        loss = mag_loss + real_loss + imag_loss

        loss = torch.mean(loss)
        return 10 * loss

    def __call__(self, batch_est, data_info):
        batch_ref = data_info[2].cuda()
        mix_len = data_info[3]
        batch, _ = batch_est[0].shape
        sisdr_lst = []
        class_label=data_info[5].cuda()
        ri_mag_compress_mse_lst=[]
        asym_lst=[]
        for i in range(batch):
            est = batch_est[0][i, :mix_len[i] * 16]
            ref = batch_ref[i, :mix_len[i] * 16]
            est_zm=est-torch.mean(est,dim=-1,keepdim=True)
            ref_zm=ref-torch.mean(ref,dim=-1,keepdim=True)
            s_ref = (self.pow_norm(est_zm, ref_zm) / (self.pow_np_norm(ref_zm) + self.eps)) * ref_zm
            e_nse = est_zm - s_ref
            sisdr_lst.append(
                10 * torch.log10((self.pow_np_norm(s_ref) / (self.pow_np_norm(e_nse) + self.eps)) + self.eps))

            target_r,target_i = self.complex_stft.transform(ref.unsqueeze(0))
            est_r,est_i = self.complex_stft.transform(est.unsqueeze(0))
            # RI_Mag_Compress_Mse(est_real, est_imag, ref_real, ref_imag, lamda=lamda)
            ri_mag_compress_mse_lst.append(self.RI_Mag_Compress_Mse(est_r,est_i,target_r,target_i,0.5))
            asym_lst.append(self.RI_Mag_Compress_Mse_ASYM(est_r,est_i,target_r,target_i,0.5))
        cosine=batch_est[1].cuda()
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0+self.eps, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, class_label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        loss = self.ce(output, class_label)
        # if min(sisdr_lst)>5:
        #     final_loss=torch.mean(torch.stack(ri_mag_compress_mse_lst))+torch.mean(torch.stack(asym_lst))-torch.mean(torch.stack(sisdr_lst))+0.5*loss
        # else:
        #     worse_case=torch.argmin(torch.stack(sisdr_lst))
        #     final_loss=ri_mag_compress_mse_lst[worse_case]+asym_lst[worse_case]-sisdr_lst[worse_case]+0.5*loss
        return torch.mean(torch.stack(ri_mag_compress_mse_lst))+torch.mean(torch.stack(asym_lst))-torch.mean(torch.stack(sisdr_lst))+0.5*loss



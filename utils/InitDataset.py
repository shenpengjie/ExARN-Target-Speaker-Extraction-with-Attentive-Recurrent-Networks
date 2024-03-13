import librosa
import numpy as np
import scipy.io as sio
import soundfile as sf
from utils.progressbar import progressbar as pb
from utils.util import makedirs, gen_list


class DatasetInitial(object):
    def __init__(self):
        self.SPEECH_LEN = 112000
        self.TR_SNR = [[-5], [-4], [-3], [-2], [-1], [0]]
        self.TT_SNR = [[-5], [0], [5]]
        self.TR_UTTERANCE_NUM = 32000
        self.CV_UTTERANCE_NUM = 3000
        self.TT_UTTERANCE_NUM = 300 * len(self.TT_SNR)

        self.TR_LST = 'tr.mat'
        self.CV_LST = 'cv.mat'
        self.CONFIG_PATH = '/data_zkh/data_environment/'
        self.TT_MIX_PATH = '/data_zkh/enhancement_data/standard_si84_test/'
        self.TT_CLEAN_PATH = '/data_zkh/enhancement_data/clean_si84_test/'
        self.TT_NOISE_PATH = '/data_zkh/enhancement_data/NOISE/'
        self.TT_NUM_SPEAKERS = 6

        self.NOISEBIN = '/data_zkh/enhancement_data/long_wav.bin'
        self.WAVBIN = '/data_zkh/enhancement_data/long_si84_one_tr_16k.bin'

        noisebin = open(self.NOISEBIN, 'rb')
        wavbin = open(self.WAVBIN, 'rb')
        noisebin.seek(0, 2)
        wavbin.seek(0, 2)
        self.NOISE_SAMPLE = noisebin.tell() / 4
        self.WAV_SAMPLE = wavbin.tell() / 4
        noisebin.close()
        wavbin.close()

    def rand_train(self, out_name='tr.mat'):
        snr_idx = np.random.randint(0, len(self.TR_SNR), size=[self.TR_UTTERANCE_NUM, 1])
        wav_idx = np.random.randint(0, self.WAV_SAMPLE - self.SPEECH_LEN, size=[self.TR_UTTERANCE_NUM, 1])
        nos_idx = np.random.randint(0, self.NOISE_SAMPLE - self.SPEECH_LEN, size=[self.TR_UTTERANCE_NUM, 1])
        mat = {'snr': self.TR_SNR, 'snr_idx': snr_idx, 'speech_idx': wav_idx, 'noise_idx': nos_idx,
               'num_utter': self.TR_UTTERANCE_NUM, 'speech_len': self.SPEECH_LEN}
        mat_path = self.CONFIG_PATH + out_name
        sio.savemat(mat_path, mat)
        return mat_path

    def rand_val(self, out_name='cv.mat'):
        snr_idx = np.random.randint(0, len(self.TR_SNR), size=[self.CV_UTTERANCE_NUM, 1])
        wav_idx = np.random.randint(0, self.WAV_SAMPLE - self.SPEECH_LEN, size=[self.CV_UTTERANCE_NUM, 1])
        nos_idx = np.random.randint(0, self.NOISE_SAMPLE - self.SPEECH_LEN, size=[self.CV_UTTERANCE_NUM, 1])
        mat = {'snr': self.TR_SNR, 'snr_idx': snr_idx, 'speech_idx': wav_idx, 'noise_idx': nos_idx,
               'num_utter': self.CV_UTTERANCE_NUM, 'speech_len': self.SPEECH_LEN}
        mat_path = self.CONFIG_PATH + out_name
        sio.savemat(mat_path, mat)
        return mat_path

    def rand_test(self):
        makedirs([self.TT_MIX_PATH])
        clean_utterances = self.readTarget(self.TT_CLEAN_PATH)
        noise_utterances = self.readTarget(self.TT_NOISE_PATH)
        num_data = round(self.TT_UTTERANCE_NUM / len(self.TT_SNR) / self.TT_NUM_SPEAKERS)

        count = 0
        pbar = pb(0, self.TT_UTTERANCE_NUM)
        pbar.start()
        for snr in self.TT_SNR:
            snr = snr[0]
            for spk_idx in range(self.TT_NUM_SPEAKERS):
                clean_idx = np.random.randint(spk_idx * 100, (spk_idx + 1) * 100, size=round(num_data / 2))
                for utt_idx in range(num_data // 2):
                    for noise_idx in range(2):
                        speech = clean_utterances[clean_idx[utt_idx]]
                        speech = speech[:len(speech) // 160 * 160]
                        noise = noise_utterances[noise_idx]
                        start_point = np.random.randint(0, len(noise) - len(speech) - 1, None)
                        noise = noise[start_point:start_point + len(speech)]
                        [mixture, nspeech, nnoise] = self.mixTargetAndNoise(speech, noise, snr)
                        MAX_V = max([max(abs(mixture)), max(abs(nspeech)), max(abs(nnoise))])
                        mixture = mixture / MAX_V
                        nspeech = nspeech / MAX_V
                        nnoise = nnoise / MAX_V
                        sf.write('%sS_%04d_%02d_%d_mix.wav' % (
                            self.TT_MIX_PATH, utt_idx + 25 * spk_idx, snr, noise_idx + 1), mixture, 16000)
                        sf.write('%sS_%04d_%02d_%d_clean.wav' % (
                            self.TT_MIX_PATH, utt_idx + 25 * spk_idx, snr, noise_idx + 1), nspeech, 16000)
                        sf.write('%sS_%04d_%02d_%d_noise.wav' % (
                            self.TT_MIX_PATH, utt_idx + 25 * spk_idx, snr, noise_idx + 1), nnoise, 16000)
                        count += 1
                        pbar.update_progress(count, 'Generating Test:')
        pbar.finish()
        return self.TT_MIX_PATH

    def readTarget(self, path):
        lst = gen_list(path, '.wav')
        rt = []
        for item in lst:
            data, fs = sf.read(path + item)
            data = librosa.resample(data, fs, 16000)
            rt.append(data)
        return rt

    def mixTargetAndNoise(self, speech, noise, snr):
        amp = np.sqrt(sum(speech ** 2) / (sum(noise ** 2) * 10 ** (float(snr) / 10.0)))
        nspeech = speech
        nnoise = noise * amp
        mixture = nspeech + nnoise
        return [mixture, nspeech, nnoise]


if __name__ == '__main__':
    Intializer = DatasetInitial()
    # tr_lst_path = Intializer.rand_train('tr_3w.mat')
    # cv_lst_path = Intializer.rand_val('cv_3w.mat')
    tt_path = Intializer.rand_test()

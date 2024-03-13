import librosa
import numpy as np
from numpy.lib.stride_tricks import as_strided
import soundfile as sf
import matplotlib.pyplot as plt
n_pad_zeros = 2
# n_fft = f_len * 2 + n_pad_zeros


def srs(s,f_len,hop_len):
    '''
    s:speech [samples,]  f_len=frame_len        hop_len=hop_size
    '''
    s = s[:len(s) // hop_len * hop_len]
    n_frames = len(s) // hop_len - 1

    if not s.flags['C_CONTIGUOUS']:
        raise ValueError('Input buffer must be contiguous.')

    frames = as_strided(s, shape=[n_frames, f_len], strides=[s.itemsize * hop_len, s.itemsize])

    # causual pad
    fft_window = librosa.filters.get_window('hann', f_len, fftbins=True)
    # fft_window = np.sqrt(fft_window)
    fft_window = librosa.util.pad_center(fft_window, f_len)
    fft_window = fft_window.reshape([1, -1])
    frames = frames * fft_window

    frames = np.pad(frames, [[0, 0], [f_len + n_pad_zeros, 0]], 'constant', constant_values=0)

    srs = np.fft.rfft(frames, axis=1)
    srs = np.real(srs)

    # srs=srs[:, 2:]

    return srs


def isrs(srs,f_len,hop_len):
    # srs=np.pad(srs, [[0,0],[2,0]],'constant', constant_values=0)
    srs = srs.astype(np.complex)
    frames = np.fft.irfft(srs)
    frames = frames[:, -f_len:]
    n_frames = frames.shape[0]
    s = np.zeros([(n_frames + 1) * hop_len])

    for i in range(n_frames):
        s[i * hop_len: i * hop_len + f_len] += frames[i]

    return s * 2


if __name__ == '__main__':
    path = "/data01/data_zkh/enhancement_data/TEST/S_0300_05_2_clean.wav"
    s1, _ = sf.read(path)
    q1= srs(s1, 320, 160)
    s1_trans = isrs(q1, 320, 160)
    result = s1_trans - s1
    plt.plot(result)
    plt.title('rs_nodc')
    plt.show()
    print("max:" + str(result.max()) + " min:" + str(result.min()))

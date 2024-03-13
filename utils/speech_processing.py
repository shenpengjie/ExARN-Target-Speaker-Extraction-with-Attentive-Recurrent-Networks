import numpy as np
def wav_segmentation(in_sig, framesamp=320, hopsamp=160, windows = 1):
    sigLength = in_sig.shape[0]
    M = (sigLength - framesamp) // hopsamp + 1
    a = np.zeros((M,framesamp))
    startpoint = 0
    for m in range(M):
        a[m, :] = in_sig[startpoint:startpoint+framesamp] * windows
        startpoint = startpoint + hopsamp
    return a
def get_early_RIR(RIR, fs):
    IR = np.copy(RIR)
    p_max = np.argmax(np.abs(IR))
    IR[:p_max - int(fs * 0.001)] = 0
    IR[p_max + int(fs * 0.05):] = 0
    return IR


def get_later_RIR(RIR, fs):
    IR = np.copy(RIR)
    p_max = np.argmax(np.abs(IR))
    IR[:p_max + int(fs * 0.05)] = 0
    return IR


def get_direct_RIR(RIR, fs):
    IR = np.copy(RIR)
    p_max = np.argmax(np.abs(IR))
    IR[:p_max] = 0
    IR[p_max + 1:] = 0
    return IR(base)
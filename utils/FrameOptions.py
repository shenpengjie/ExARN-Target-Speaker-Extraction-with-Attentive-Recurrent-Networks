import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
class TorchChunking(nn.Module):
    def __init__(self, frame_size=512, frame_shift=256, pad_left=0):
        super(TorchChunking, self).__init__()
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.pad_left = pad_left
        self.kernel_size = self.frame_size
        self.stride = self.frame_shift

    def get_frames(self, in_sig):
        N = in_sig.shape[-1]
        N_new = N + self.pad_left
        pad_right = (
            (N_new // self.frame_shift - 1) * self.frame_shift + self.frame_size - N_new
        )
        out = F.pad(in_sig, (self.pad_left, pad_right))
        # out = torch.unsqueeze(out, dim=-1)
        out =out.unfold(1,self.kernel_size,self.stride)
        out = out.transpose(-2, -1)
        return out

    def ola(self, inputs, size):
        inputs = inputs.transpose(-2, -1)
        den = torch.ones_like(inputs)
        N_new = size + self.pad_left
        pad_right = (
            (N_new // self.frame_shift - 1) * self.frame_shift + self.frame_size - N_new
        )
        out = F.fold(
            inputs,
            output_size=(size + self.pad_left + pad_right, 1),
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=(0, 0),
        )

        den = F.fold(
            den,
            output_size=(size + self.pad_left + pad_right, 1),
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=(0, 0),
        )
        out = out / den
        out = out.squeeze(dim=-1)
        return out[..., self.pad_left : self.pad_left + size]

class TorchOLA(nn.Module):
    r"""Overlap and add on gpu using torch tensor"""

    # Expects signal at last dimension
    def __init__(self, frame_shift=256):
        super(TorchOLA, self).__init__()
        self.frame_shift = frame_shift

    def forward(self, inputs):
        nframes = inputs.shape[-2]
        frame_size = inputs.shape[-1]
        frame_step = self.frame_shift
        sig_length = (nframes - 1) * frame_step + frame_size
        sig = torch.zeros(list(inputs.shape[:-2]) + [sig_length], dtype=inputs.dtype, device=inputs.device,
                          requires_grad=False)
        ones = torch.zeros_like(sig)
        start = 0
        end = start + frame_size
        for i in range(nframes):
            sig[..., start:end] += inputs[..., i, :]
            ones[..., start:end] += 1.
            start = start + frame_step
            end = start + frame_size
        return sig / ones


class TorchSignalToFrames(object):
    def __init__(self, frame_size=512, frame_shift=256):
        super(TorchSignalToFrames, self).__init__()
        self.frame_size = frame_size
        self.frame_shift = frame_shift

    def __call__(self, in_sig):
        sig_len = in_sig.shape[-1]
        nframes = (sig_len // self.frame_shift)
        a = torch.zeros(tuple(in_sig.shape[:-1]) + (nframes, self.frame_size), device=in_sig.device)
        start = 0
        end = start + self.frame_size
        k = 0
        for i in range(nframes):
            if end < sig_len:
                a[..., i, :] = in_sig[..., start:end]
                k += 1
            else:
                tail_size = sig_len - start
                a[..., i, :tail_size] = in_sig[..., start:]

            start = start + self.frame_shift
            end = start + self.frame_size
        return a


def pad_input(input, window):
    """
    Zero-padding input according to window/stride size.
    """
    batch_size, nsample = input.shape
    stride = window // 2

    # pad the signals at the end for matching the window/stride size
    rest = window - (stride + nsample % window) % window
    if rest > 0:
        pad = torch.zeros(batch_size, rest).type(input.type())
        input = torch.cat([input, pad], 1)
    pad_aux = torch.zeros(batch_size, stride).type(input.type())
    input = torch.cat([pad_aux, input, pad_aux], 1)

    return input, rest


def frames_data(noisy_speech, hop_size=160, use_window=True):
    noisy_speech = noisy_speech
    hamming_win = torch.from_numpy(np.hamming(hop_size * 2).astype(np.float32)).cuda().reshape(1, 1, hop_size * 2)

    # noisy_speech = [batch_size, nsample]
    noisy_speech = noisy_speech[:, :noisy_speech.size(1) // hop_size * hop_size]
    # left data
    frame_speech = noisy_speech.reshape(noisy_speech.size(0), -1, hop_size)
    # right data
    frame_speech2 = frame_speech.clone()[:, 1:]

    frames = torch.cat([frame_speech[:, :-1], frame_speech2[:, :]], dim=-1)
    if use_window is True:
        frames = frames * hamming_win
    return frames


def overlap_data(data, hop_size=160, deweight=True):
    left_data = data[:, :, :hop_size]
    left_data = torch.cat([left_data, torch.zeros_like(left_data[:, -1:, :]).cuda()], dim=1)
    right_data = data[:, :, hop_size:]
    right_data = torch.cat([torch.zeros_like(right_data[:, 0:1, :]).cuda(), right_data], dim=1)

    # [:,:left_data.size(1), :]
    overlap_res = (left_data + right_data).reshape(data.size(0), -1)
    if deweight is True:
        overlap_res[..., hop_size:-hop_size] /= 2.0
    return overlap_res


def chunks_data(noisy_speech, chunk_size=154, chunk_shift=77):
    [B, T, L] = noisy_speech.shape
    # how many chunks
    J = np.ceil(T / chunk_shift)
    # pad zeros
    dis = (J - 1) * chunk_shift + chunk_size - T
    noisy_speech = F.pad(noisy_speech, [0, 0, 0, int(dis)], mode='constant', value=0)
    # chunks
    left_data = noisy_speech.reshape(B, -1, chunk_shift, L)
    right_data = left_data.clone()
    chunks = torch.cat([left_data[:, :-1, :, :], right_data[:, 1:, :, :]], dim=2)
    return chunks, dis


def chunks_overlap(data, chunk_shift=77):
    left_data = data[:, :, :chunk_shift, :]
    left_data = torch.cat([left_data, torch.zeros_like(left_data[:, -1:, :, :]).cuda()], dim=1)
    right_data = data[:, :, chunk_shift:, :]
    right_data = torch.cat([torch.zeros_like(right_data[:, 0:1, :, :]).cuda(), right_data], dim=1)

    # [:,:left_data.size(1), :]
    overlap_res = (left_data + right_data).reshape(data.size(0), -1, data.size(3))
    return overlap_res

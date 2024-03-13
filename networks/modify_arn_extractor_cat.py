import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile, clever_format
from torch.autograd import Variable
from torch.nn import init

from components.arn_wrapper import arn_block_separate_cat
from components.res_block import ResBlock,ECAPA
from utils.FrameOptions import TorchSignalToFrames, TorchOLA,TorchChunking

class GlobalLayerNorm(nn.Module):
    """
    Global channel layer normalization
    """

    def __init__(self, dim, eps=1e-05, elementwise_affine=True):
        super(GlobalLayerNorm, self).__init__()
        self.eps = eps
        self.normalized_dim = dim
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.beta = nn.Parameter(torch.zeros(1,dim))
            self.gamma = nn.Parameter(torch.ones(1,dim))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        """
        x: N x C x T
        """
        if x.dim() != 2:
            raise RuntimeError("{} accept 2D tensor as input".format(
                self.__name__))
        # N x 1 x 1
        mean = torch.mean(x, (0, 1), keepdim=True)
        var = torch.mean((x - mean)**2, (0, 1), keepdim=True)
        # N x T x C
        if self.elementwise_affine:
            x = self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta
        else:
            x = (x - mean) / torch.sqrt(var + self.eps)
        return x

    def extra_repr(self):
        return "{normalized_dim}, eps={eps}, " \
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)


class NET_Wrapper(nn.Module):
    def __init__(self, frame_size, frame_shift, feature_dim, hidden_dim,infer=False, causal=True):
        super(NET_Wrapper, self).__init__()
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.feature_dim = feature_dim
        self.infer = infer
        self.causal = causal
        self.kernel_size = (self.frame_size, 1)
        self.stride = (self.frame_shift, 1)
        # self.chunking = TorchChunking(
        #     frame_size=self.frame_size,
        #     frame_shift=self.frame_shift,
        #     pad_left=0,
        # )
        # self.get_frames = TorchSignalToFrames(frame_size, frame_shift)
        # self.get_signal = TorchOLA(frame_shift)
        self.input_proj = nn.Linear(self.frame_size, self.feature_dim)
        self.output_proj = nn.Linear(self.feature_dim, self.frame_size,bias=True)

        self.aux_layer = GlobalLayerNorm(feature_dim)
        self.res_bloccks=nn.Sequential(
            ECAPA(self.feature_dim,hidden_dim)
        )

        self.attention_layer=nn.Sequential(
            nn.Conv1d(hidden_dim*3, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Tanh(), # I add this layer
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.Softmax(dim=2)
                                          )
        self.classify=nn.Linear(in_features=self.feature_dim,out_features=101)

        self.bn5 = nn.BatchNorm1d(hidden_dim*2)
        self.fc6 = nn.Conv1d(hidden_dim*2, self.feature_dim,kernel_size=1)
        self.bn6 = nn.BatchNorm1d(self.feature_dim)
        self.arn_block1 = arn_block_separate_cat(self.feature_dim, infer=self.infer, causal=self.causal)
        self.arn_block2 = arn_block_separate_cat(self.feature_dim, infer=self.infer, causal=self.causal)
        self.arn_block3 = arn_block_separate_cat(self.feature_dim, infer=self.infer, causal=self.causal)
        self.arn_block4 = arn_block_separate_cat(self.feature_dim, infer=self.infer, causal=self.causal)
        # self.initialize_params()

    def forward(self, input,anchor,input_len,aux_len):
        # chunk frames

        batch,len_sig = input.shape
        # frames = self.chunking.get_frames(input)
        # aux_frames=self.chunking.get_frames(anchor)
        # frames = self.get_frames(input)
        # aux_frames=self.get_frames(anchor)
        frames = self.get_frames(input.unsqueeze(1))
        aux_frames = self.get_frames(anchor.unsqueeze(1))

        out=self.input_proj(frames)
        aux_out=self.input_proj(aux_frames)

        aux_gln_lst=[]
        for i in range(batch):
            one_aux=aux_out[i,:aux_len[i],:]
            aux_gln_lst.append(self.aux_layer(one_aux))
        aux_layer_out=torch.nn.utils.rnn.pad_sequence(aux_gln_lst,batch_first=True)
        # aux_out=self.aux_layer(aux_out)
        aux_layer_out=aux_layer_out.transpose(1,2)
        aux_out = self.res_bloccks(aux_layer_out)
        aux_emb_list=[]
        for i in range(batch):
            one_aux=aux_out[i,:,:aux_len[i]].unsqueeze(0)
            global_one_aux= torch.cat((aux_out[i,:,:aux_len[i]].unsqueeze(0),torch.mean(one_aux,dim=2,keepdim=True).repeat(1,1,aux_len[i]), torch.sqrt(torch.var(one_aux,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,aux_len[i])), dim=1)
            w = self.attention_layer(global_one_aux)
            mu = torch.sum(one_aux * w, dim=2, keepdim=True)
            sg = torch.sqrt((torch.sum((one_aux ** 2) * w, dim=2, keepdim=True) - mu ** 2).clamp(min=1e-4))
            x = torch.cat((mu, sg), 1)
            aux_emb_list.append(x)
        aux_emb=torch.stack(aux_emb_list,dim=0).squeeze(1)

        aux_emb = self.bn5(aux_emb)
        aux_emb = self.fc6(aux_emb)
        aux_emb = self.bn6(aux_emb)
        aux_emb=aux_emb.permute(0,2,1)
        # input linear proj
        # out = self.input_proj(frames)
        # main body : 4 * ARN Block
        classify_out=self.classify(torch.nn.functional.normalize(aux_emb.squeeze(1)))
        out = self.arn_block1(out,aux_emb,input_len)
        out = self.arn_block2(out,aux_emb,input_len)
        out = self.arn_block3(out,aux_emb,input_len)
        out = self.arn_block4(out,aux_emb,input_len)
        # output linear proj
        out = self.output_proj(out)
        # overlap and add
        # signal = self.chunking.ola(out,len_sig)
        # signal = self.get_signal(out)[:,:len_sig]
        signal = self.ola(out, len_sig)
        signal = signal.squeeze(1)

        return signal,classify_out

    def get_frames(self, in_sig):
        N = in_sig.shape[-1]

        pad_right = (
            (N // self.frame_shift - 1) * self.frame_shift + self.frame_size - N
        )
        out = F.pad(in_sig, (0, pad_right))
        out = torch.unsqueeze(out, dim=-1)
        out = F.unfold(
            out, kernel_size=self.kernel_size, stride=self.stride, padding=(0, 0)
        )
        out = out.transpose(-2, -1)
        return out

    def ola(self, inputs, size):
        inputs = inputs.transpose(-2, -1)
        den = torch.ones_like(inputs)

        pad_right = (
            (size // self.frame_shift - 1) * self.frame_shift + self.frame_size - size
        )
        out = F.fold(
            inputs,
            output_size=(size  + pad_right, 1),
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=(0, 0),
        )

        den = F.fold(
            den,
            output_size=(size + pad_right, 1),
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=(0, 0),
        )
        out = out / den
        out = out.squeeze(dim=-1)
        return out[..., 0 : size]


if __name__ == '__main__':
    from ptflops import get_model_complexity_info
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    input = Variable(torch.FloatTensor(torch.rand(4, 32000))).cuda()
    anchor=Variable(torch.FloatTensor(torch.rand(4, 32000))).cuda()
    input_length=torch.tensor([2000,999,888,777])
    anchor_length=torch.tensor([698,364,254,1000])
    net = NET_Wrapper(128, 16, 512, 512, False, False).cuda()
    # macs, params = get_model_complexity_info(net, [(16000,),(16000,),(800,),(800,)], as_strings=True, print_per_layer_stat=True)
    out=net(input,anchor,input_length,anchor_length)

    parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(parameters)
    # macs, params = profile(net, inputs=(input,anchor,))
    # macs, params = clever_format([macs, params], "%.3f")
    # print(macs)
    # print(params)
    # print("%s | %.2f | %.2f" % ('elephantstudent', params, macs))

import os

import yaml
import soundfile as sf
from dataloader import SpeechMixDataset, BatchDataLoader


def gen_cv_wavs(output_path):

    with open('../configs/ARN_nc.yaml', 'r') as f_yaml:
        config = yaml.load(f_yaml, Loader=yaml.FullLoader)
    cv_mix_dataset = SpeechMixDataset(config, mode='validate')
    cv_batch_dataloader = BatchDataLoader(cv_mix_dataset, config['BATCH_SIZE'], is_shuffle=False,
                                          workers_num=config['NUM_WORK'])

    for j, batch_eval in enumerate(cv_batch_dataloader.get_dataloader()):
        [mixture, clean, noise, wav_name] = batch_eval
        batch, _ = mixture.shape
        for b in range(batch):
            sf.write(os.path.join(output_path, 'mix.wav'), mixture[b], config['SAMPLE_RATE'])
            sf.write(os.path.join(output_path, 'clean.wav'), mixture[b], config['SAMPLE_RATE'])


if __name__ == '__main__':
    gen_cv_wavs('/mnt/raid2/user_space/zhangkanghao/data/LIBRI_datasets/libri_test_clips')
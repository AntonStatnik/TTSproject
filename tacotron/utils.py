import torch
import argparse
import numpy as np
import matplotlib.pylab as plt
import time, gc
from model import Tacotron2, mode, to_arr
from dataset import hparams as hps, save_wav, inv_melspectrogram, text_to_sequence
from tensorboardX import SummaryWriter


def mode(obj, model=False):
    if model and hps.is_cuda:
        obj = obj.cuda()
    elif hps.is_cuda:
        obj = obj.cuda(non_blocking=hps.pin_mem)
    return obj

def to_arr(var):
    return var.cpu().detach().numpy().astype(np.float32)

def get_mask_from_lengths(lengths, pad=False):
    max_len = torch.max(lengths).item()
    if pad and max_len % hps.n_frames_per_step != 0:
        max_len += hps.n_frames_per_step - max_len % hps.n_frames_per_step
        assert max_len % hps.n_frames_per_step == 0
    ids = torch.arange(0, max_len, out=torch.LongTensor(max_len))
    ids = mode(ids)
    mask = (ids < lengths.unsqueeze(1))
    return mask


def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data.transpose(2, 0, 1)

def plot_alignment_to_numpy(alignment, info=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment, aspect='auto', origin='lower', interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data

def plot_spectrogram_to_numpy(spectrogram):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


class Tacotron2Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Tacotron2Logger, self).__init__(logdir, flush_secs = 5)

    def log_training(self, items, grad_norm, learning_rate, iteration):
            self.add_scalar('loss.mel', items[0], iteration)
            self.add_scalar('loss.gate', items[1], iteration)
            self.add_scalar('grad.norm', grad_norm, iteration)
            self.add_scalar('learning.rate', learning_rate, iteration)

    def sample_train(self, outputs, iteration):
            mel_outputs = to_arr(outputs[0][0])
            mel_outputs_postnet = to_arr(outputs[1][0])
            alignments = to_arr(outputs[3][0]).T
            
            # plot alignment, mel and postnet output
            self.add_image(
                'train.align',
                plot_alignment_to_numpy(alignments),
                iteration)
            self.add_image(
                'train.mel',
                plot_spectrogram_to_numpy(mel_outputs),
                iteration)
            self.add_image(
                'train.mel_post',
                plot_spectrogram_to_numpy(mel_outputs_postnet),
                iteration)

    def sample_infer(self, outputs, iteration):
            mel_outputs = to_arr(outputs[0][0])
            mel_outputs_postnet = to_arr(outputs[1][0])
            alignments = to_arr(outputs[2][0]).T
            
            # plot alignment, mel and postnet output
            self.add_image(
                'infer.align',
                plot_alignment_to_numpy(alignments),
                iteration)
            self.add_image(
                'infer.mel',
                plot_spectrogram_to_numpy(mel_outputs),
                iteration)
            self.add_image(
                'infer.mel_post',
                plot_spectrogram_to_numpy(mel_outputs_postnet),
                iteration)
            
            # save audio
            wav = inv_melspectrogram(mel_outputs)
            wav_postnet = inv_melspectrogram(mel_outputs_postnet)
            self.add_audio('infer.wav', wav, iteration, hps.sample_rate)
            self.add_audio('infer.wav_post', wav_postnet, iteration, hps.sample_rate)


# +
# Timing utilities
start_time = None

def start_timer():
    global start_time
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    start_time = time.time()

def end_timer_and_print(local_msg):
    torch.cuda.synchronize()
    end_time = time.time()
    print("\n" + local_msg)
    print("Total execution time = {:.3f} sec".format(end_time - start_time))
    print("Max memory used by tensors = {} bytes".format(torch.cuda.max_memory_allocated()))

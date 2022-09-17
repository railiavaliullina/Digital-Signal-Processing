import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as sw
from scipy.signal import hilbert

from config import cfg


def make_plot(x, title):
    plt.figure(figsize=(10, 6))
    plt.plot(x)
    plt.title(title)
    if cfg.save_all_plots:
        print(f'Saving plot {title}...')
        plt.savefig(f'{title}.png')
        plt.clf()
    else:
        plt.show()


def read_wav_file(path):
    fs, x = sw.read(path)
    return fs, x


def get_inst_frequency(x, fs):
    # мгновен. частота
    z = hilbert(x)
    w_t = np.diff(np.unwrap(np.angle(z)))
    w_t_scaled = fs / (2 * np.pi) * w_t
    return w_t_scaled


def get_sequence_of_bits():
    thr_for_ones = cfg.one_freq - (cfg.one_freq - cfg.base_freq) / 2
    thr_for_zeros = cfg.zero_freq + (cfg.base_freq - cfg.zero_freq) / 2

    ones_ids = np.where(w_t_scaled > thr_for_ones)[0]
    zero_ids = np.where(w_t_scaled < thr_for_zeros)[0]
    base_ids = np.where(np.logical_and(w_t_scaled <= thr_for_ones, w_t_scaled >= thr_for_zeros))[0]
    assert len(w_t_scaled) == len(ones_ids) + len(zero_ids) + len(base_ids)

    zeros_seq_differences = zero_ids[1:] - zero_ids[:-1]
    thresholds_ids_for_zero = np.where(zeros_seq_differences > 1)[0]

    ones_seq_differences = ones_ids[1:] - ones_ids[:-1]
    thresholds_ids_for_one = np.where(ones_seq_differences > 1)[0]

    base_seq_differences = base_ids[1:] - base_ids[:-1]
    thresholds_ids_for_base = np.where(base_seq_differences > 1)[0]

    thr_zeros = np.insert(zero_ids[thresholds_ids_for_zero], -1, zero_ids[-1])
    thr_ones = np.insert(ones_ids[thresholds_ids_for_one], -1, ones_ids[-1])
    thr_base = np.insert(base_ids[thresholds_ids_for_base], -1, base_ids[-1])

    united_ids = list(thr_zeros) + list(thr_ones) + list(thr_base)
    united_ids_labels = list(np.zeros(len(thr_zeros))) + list(np.ones(len(thr_ones))) + list(
        np.ones(len(thr_base)) * -1)
    united_ids_sorted_args = np.argsort(united_ids)
    united_ids_labels_sorted = np.asarray(united_ids_labels)[united_ids_sorted_args]

    bits_sequence = united_ids_labels_sorted[united_ids_labels_sorted > -1].astype(int)
    print(f'Sequence of bits: {bits_sequence}')


if __name__ == '__main__':

    fs, x = read_wav_file(cfg.path)
    # мгновен. частота
    w_t_scaled = get_inst_frequency(x, fs)
    make_plot(w_t_scaled, 'Instantaneous frequency')
    get_sequence_of_bits()

import numpy as np
import scipy.io.wavfile as sw
from matplotlib import pyplot as plt
from numpy.lib import stride_tricks

from config import cfg


def get_matrix(sig):
    w, p = np.hanning(cfg.w_size), cfg.w_size - np.floor(0.5 * cfg.w_size)
    rows = np.append(np.zeros(int(np.floor(cfg.w_size/2.0))), sig)
    cols = np.ceil((len(rows) - cfg.w_size) / p) + 1
    rows = np.append(rows, np.zeros(cfg.w_size))
    s = rows.strides[0]
    matrix = stride_tricks.as_strided(rows, shape=(int(cols), cfg.w_size), strides=(s*int(p), s))
    multiplied_by_win = matrix * w
    m_ = []
    for i in range(len(multiplied_by_win)//cfg.w_size2):
        m_.append(multiplied_by_win[i*cfg.w_size2:(i+1)*cfg.w_size2])
    m_ = np.reshape(np.stack(m_), (-1, cfg.w_size))
    fin_matrix = np.fft.rfft(m_)  # multiplied_by_win
    return np.transpose(fin_matrix)


def plot_spectrogram_matrix(samples, type, file_name):
    matrix = get_matrix(samples)
    matrix = np.log10(np.abs(matrix))
    matrix = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))
    matrix *= 255

    # plot spectrogram
    plt.figure(figsize=(10, 6))
    plt.imshow(matrix, origin="lower", aspect="auto", cmap="jet")
    plt.title(f'{type}\n{file_name}')
    # plt.show()
    plt.savefig(f'plots/{type}_{file_name}.png')
    plt.clf()

    # plot f0
    if type == 'voice':
        F0 = np.argmax(matrix, 0)
        plt.plot(np.arange(len(F0)), F0)
        plt.title(f'F0_{file_name}')
        # plt.show()
        plt.savefig(f'plots/F0_{file_name}.png')
        plt.clf()


if __name__ == '__main__':
    # голос1 (voice.wav)
    fs1, voice_signal1 = sw.read(cfg.path1)
    voice1 = voice_signal1[10000:80000]
    silence1 = voice_signal1[38000:65500]  #  61390:107380
    plot_spectrogram_matrix(voice1, type='voice', file_name='voice.wav')
    plot_spectrogram_matrix(silence1, type='silence', file_name='voice.wav')

    # голос2 (voice2.wav)
    fs2, voice_signal2 = sw.read(cfg.path2)
    voice2 = voice_signal2[3000:58000]
    silence2 = voice_signal2[50000:68000]
    plot_spectrogram_matrix(voice2, type='voice', file_name='voice2.wav')
    plot_spectrogram_matrix(silence2, type='silence', file_name='voice2.wav')

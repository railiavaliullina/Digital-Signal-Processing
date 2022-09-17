import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sgn
from scipy.fft import rfft, rfftfreq


def make_plot(x, title, save=False, folder_to_save=''):
    plt.figure(figsize=(10, 6))
    if isinstance(x, tuple):
        x, y = x
        plt.plot(x, y)
    else:
        plt.plot(x)
    plt.title(title)
    if save:
        print(f'Saving plot {title} to plots/{folder_to_save}/ dir...')
        plt.savefig(f'plots/{folder_to_save}/{title}.png')
        plt.clf()
    else:
        plt.show()


# 15Гц, 30Гц, 50 Гц, 75Гц, 90Гц
def get_signal():
    t = np.linspace(0, time, time * fs)
    signals = []
    for F in [15, 30, 50, 75, 90]:
        s = np.sin(2 * np.pi * F * t)
        signals.append(s)

        yf_orig = rfft(s)
        xf_orig = rfftfreq(len(s), 1 / fs)
        # make_plot((xf_orig, np.abs(yf_orig)), f'spectrum of signal with F={F}')

    out = np.sum(signals, 0)
    make_plot(out, f'Summarized signal', save=save_plots, folder_to_save='two_bandpass_filters')
    make_plot(out, f'Summarized signal', save=save_plots, folder_to_save='bandpass_and_bandstop_filters')
    yf_out = rfft(out)
    xf_out = rfftfreq(len(out), 1 / fs)
    make_plot((xf_out, np.abs(yf_out)), f'Spectrum of summarized signal', save=save_plots,
              folder_to_save='two_bandpass_filters')
    make_plot((xf_out, np.abs(yf_out)), f'Spectrum of summarized signal', save=save_plots,
              folder_to_save='bandpass_and_bandstop_filters')
    return out


def apply_two_bandpass_filters(s):
    b1, a1 = sgn.butter(order1, [0.29, 0.31], btype='bandpass')
    w2, h2 = sgn.freqz(b1, a1)
    make_plot((w2 / np.pi, abs(h2)), f'Transmission function of IIR filter (F={cutoff_freq1})',
              save=save_plots, folder_to_save='two_bandpass_filters')

    freqs = np.arange(0, 1.05, 0.05)
    gain = np.zeros(len(freqs))
    gain[15] = 1
    b2 = sgn.firwin2(order2, freqs, gain)
    a2 = 1
    w3, h3 = sgn.freqz(b2, a2)
    make_plot((w3 / np.pi, abs(h3)), f'Transmission function of FIR filter (F={cutoff_freq2})',
              save=save_plots, folder_to_save='two_bandpass_filters')

    b = np.convolve(b1, b2, 'full')
    a = np.convolve(a1, a2, 'full')
    w, h = sgn.freqz(b, a)
    make_plot((w / np.pi, abs(h)), f'Transmission function of result filter', save=save_plots,
              folder_to_save='two_bandpass_filters')

    filtered_signal = sgn.filtfilt(b, a, s)
    make_plot(filtered_signal, f'Filtered signal', save=save_plots, folder_to_save='two_bandpass_filters')
    yf_out = rfft(filtered_signal)
    xf_out = rfftfreq(len(filtered_signal), 1 / fs)
    make_plot((xf_out, np.abs(yf_out)), f'Spectrum of filtered signal', save=save_plots,
              folder_to_save='two_bandpass_filters')


def apply_bandpass_and_bandstop_filters(s):
    b1, a1 = sgn.butter(order1, [0.31, 0.74], btype='bandstop')
    w1, h1 = sgn.freqz(b1, a1)
    make_plot((w1 / np.pi, abs(h1)), f'Transmission function of IIR filter (bandstop for F=[{cutoff_freq1 + 1}, {cutoff_freq2 - 1}])',
              save=save_plots, folder_to_save='bandpass_and_bandstop_filters')

    b2 = sgn.firwin(order2, [0.29, 0.76], pass_zero='bandpass')
    a2 = 1
    w2, h2 = sgn.freqz(b2, a2)
    make_plot((w2 / np.pi, abs(h2)), f'Transmission function of FIR filter (bandpass for F=[{cutoff_freq1 - 1}, {cutoff_freq2 + 1}])',
              save=save_plots, folder_to_save='bandpass_and_bandstop_filters')

    b = np.convolve(b1, b2, 'full')
    a = np.convolve(a1, a2, 'full')
    w, h = sgn.freqz(b, a)
    make_plot((w / np.pi, abs(h)), f'Transmission function of result filter', save=save_plots, folder_to_save='bandpass_and_bandstop_filters')

    filtered_signal = sgn.filtfilt(b, a, s)
    make_plot(filtered_signal, f'Filtered signal', save=save_plots, folder_to_save='bandpass_and_bandstop_filters')
    yf_out = rfft(filtered_signal)
    xf_out = rfftfreq(len(filtered_signal), 1 / fs)
    make_plot((xf_out, np.abs(yf_out)), f'Spectrum of filtered signal', save=save_plots, folder_to_save='bandpass_and_bandstop_filters')


if __name__ == '__main__':
    fs = 200
    time = 10
    order1 = 2
    order2 = 65
    cutoff_freq1 = 30
    cutoff_freq2 = 75
    save_plots = True

    s = get_signal()
    # вариант с объединением bandpass IIR и bandpass FIR фильтров
    print(f'\nFiltering with bandpass IIR and FIR filters...')
    apply_two_bandpass_filters(s)

    # вариант с объединением bandstop IIR фильтра и bandpass FIR фильтров
    print(f'\nFiltering with bandstop IIR and bandpass FIR filters...')
    apply_bandpass_and_bandstop_filters(s)

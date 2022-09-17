import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sgn
import scipy.io.wavfile as sw
from scipy.fft import rfft, rfftfreq, fft, fftfreq


def plot_graphs(spectrum, shifted_spectrum, plots_names, title='', to_save=False, path_to_save=''):
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    axs = axs.flatten()
    plots_args = [spectrum, shifted_spectrum]
    plt.title(title)
    fig.tight_layout(pad=4.5)
    for i, ax in enumerate(axs):
        ax.plot(plots_args[i][0], plots_args[i][1])
        ax.set_title(title + plots_names[i], fontsize=10, fontweight="bold")
    if to_save:
        plt.savefig(path_to_save)
    else:
        plt.show()


def make_plot(x, y, title):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.title(title)
    if save_all_plots:
        print(f'Saving plot {title} to plots/ dir...')
        plt.savefig(f'plots/{title}.png')
        plt.clf()
    else:
        plt.show()


if __name__ == '__main__':
    w = 0.03
    order = 13
    k = 2
    path = 'voice.wav'
    save_all_plots = True

    fs, x = sw.read(path)
    yf_orig = rfft(x)
    xf_orig = rfftfreq(len(x), 1 / fs)
    make_plot(xf_orig, np.abs(yf_orig), f'Original spectrum')

    freq_band = 500  # ширина окна
    assert w * fs > freq_band / 2, f'invalid freq_band'
    nb_bands = fs // (2 * freq_band)

    freq_values_per_band = np.arange(0, fs // 2).reshape(nb_bands, -1)
    sig_with_applied_filters = []
    yf_with_applied_filters = []
    is_zero, is_zero1 = False, False

    for band_id, band_freq_values in enumerate(freq_values_per_band):

        if band_id in [0, len(freq_values_per_band) - 1]:
            # low/high pass filters
            filter_type = 'lowpass' if band_id == 0 else 'highpass'
            cutoff = np.max(band_freq_values) if filter_type == 'lowpass' else np.min(band_freq_values)
            normal_cutoff = cutoff / (float(fs) / 2)
            # b, a = sgn.butter(order, normal_cutoff, btype=filter_type)
            b = sgn.firwin(order, normal_cutoff, pass_zero=filter_type)
            a = 1
            filtered_signal = sgn.filtfilt(b, a, x)
            title_name = filter_type + ' filter'
        else:
            # band filters
            if len(np.where(np.logical_and(xf_orig > np.min(band_freq_values), xf_orig < np.max(band_freq_values)))[0]) > 0:
                normal_cutoff = (np.min(band_freq_values) / (float(fs) / 2), np.max(band_freq_values) / (float(fs) / 2))
                # b, a = sgn.butter(order, normal_cutoff, btype='bandpass')
                b = sgn.firwin(order, normal_cutoff, pass_zero='bandpass')
                a = 1
                filtered_signal = sgn.filtfilt(b, a, x)
                if np.isnan(filtered_signal).any():
                    filtered_signal = np.zeros(len(filtered_signal))
                    is_zero = True
                title_name = f'bandpass filter #{band_id}'

        w_trans_f, h_trans_f = sgn.freqz(b)
        make_plot(w_trans_f / np.pi, abs(h_trans_f),
                  f'Transmission function of {title_name}')

        assert not np.isnan(filtered_signal).any()
        filtered_yf = rfft(filtered_signal)
        filtered_xf = rfftfreq(len(filtered_signal), 1 / fs)
        if not is_zero:
            make_plot(filtered_xf, np.abs(filtered_yf), f'Spectrum after applying {title_name}')

        cos = np.cos(2 * np.pi * w * np.arange(len(filtered_signal)))
        shifted_filtered_signal = filtered_signal * cos
        assert not np.isnan(shifted_filtered_signal).any()
        shifted_filtered_yf = rfft(shifted_filtered_signal)
        shifted_filtered_xf = rfftfreq(len(shifted_filtered_signal), 1 / fs)
        if not is_zero:
            make_plot(shifted_filtered_xf, np.abs(shifted_filtered_yf), f'Spectrum after applying {title_name} and shifting')

        # high pass filter
        cutoff = np.max(band_freq_values)
        normal_cutoff = cutoff / (float(fs) / 2)
        b, a = sgn.butter(order, normal_cutoff, btype='high')
        shifted_filtered_signal_with_no_dup = sgn.filtfilt(b, a, shifted_filtered_signal)
        if np.isnan(shifted_filtered_signal_with_no_dup).any():
            shifted_filtered_signal_with_no_dup = np.zeros(len(shifted_filtered_signal_with_no_dup))
            is_zero1 = True
        assert not np.isnan(shifted_filtered_signal_with_no_dup).any()
        shifted_filtered_yf_with_no_dup = rfft(shifted_filtered_signal_with_no_dup)
        shifted_filtered_xf_with_no_dup = rfftfreq(len(shifted_filtered_signal_with_no_dup), 1 / fs)
        if not is_zero1:
            make_plot(shifted_filtered_xf_with_no_dup, np.abs(shifted_filtered_yf_with_no_dup),
                      f'Spectrum after applying {title_name} and shifting (no duplicate)')

        sig_with_applied_filters.append(shifted_filtered_signal_with_no_dup)
        yf_with_applied_filters.append(shifted_filtered_yf_with_no_dup)
        w_trans_f, h_trans_f = sgn.freqz(b)
        make_plot(w_trans_f / np.pi, abs(h_trans_f), f'Transmission function of high pass filter (duplicates remove)')

        is_zero = False

    xf_out = shifted_filtered_xf_with_no_dup
    yf_out = np.sum(np.asarray(yf_with_applied_filters), 0)
    # make_plot(xf_out, np.abs(yf_out), f'Spectrum of out signal')

    sig_out = np.sum(np.asarray(sig_with_applied_filters), 0)
    sig_out_scaled = k * sig_out
    yf_out_scaled = rfft(sig_out_scaled)
    make_plot(xf_out, np.abs(yf_out_scaled), f'Spectrum of out signal after scaling')

    plot_graphs((xf_orig, np.abs(yf_orig)), (xf_out, np.abs(yf_out_scaled)),
                plots_names=['original spectrum', 'shifted spectrum'],
                to_save=save_all_plots, path_to_save='plots/original_and_shifted_spectrum.png')

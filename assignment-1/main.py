import numpy as np
import scipy.io.wavfile as sw
import bisect

from config import cfg
from utils import plot_all_hists_for_signal


def get_sine_signal(F, fs, time):
    t = np.linspace(0, time, time * fs)
    s = np.sin(2 * np.pi * F * t)
    return s, t


def get_signal_with_uniform_dist(fs, time):
    uniform_dist_signal = np.random.uniform(low=-1, high=1, size=fs * time)
    return uniform_dist_signal


def get_signal_interval(signal):
    signal_to_quantize = signal[:cfg.nb_values_to_quantize]
    min_v, max_v = np.min(signal_to_quantize), np.max(signal_to_quantize)
    interval_step = (max_v - min_v) / (cfg.nb_intervals - 1)
    intervals = np.arange(min_v, max_v + interval_step, interval_step)
    return intervals, signal_to_quantize


def get_quantized_signal(signal):
    intervals, signal_to_quantize = get_signal_interval(signal)
    quantized_signal = []
    for signal_value in signal_to_quantize:
        upper_boundary_idx = bisect.bisect_left(intervals, signal_value)
        upper_boundary_idx = np.min([upper_boundary_idx, len(intervals) - 1])
        upper_boundary = intervals[upper_boundary_idx]
        lower_boundary = intervals[upper_boundary_idx - 1]
        rounded_signal_value = min([upper_boundary, lower_boundary], key=lambda x: abs(x - signal_value))
        quantized_signal.append(rounded_signal_value)
    quantized_signal = np.asarray(quantized_signal)
    error = abs(signal_to_quantize - quantized_signal)
    return np.asarray(quantized_signal), error, signal_to_quantize, intervals


if __name__ == '__main__':

    # синусоидальный сигнал
    sine_signal, t = get_sine_signal(F=10, fs=16000, time=10)
    quantized_sine_signal, error, signal_to_quantize, _ = get_quantized_signal(sine_signal)
    snr_sine_signal = 10 * np.log10(np.var(signal_to_quantize)/np.var(error))
    print(f'Sine signal SNR: {snr_sine_signal}')
    plot_all_hists_for_signal(signal_to_quantize, quantized_sine_signal, error, signal_name='Sine')

    # сигнал с равномерным распределением
    uniform_signal = get_signal_with_uniform_dist(fs=16000, time=10)
    quantized_uniform_signal, error, signal_to_quantize, _ = get_quantized_signal(uniform_signal)
    snr_uniform_signal = 10 * np.log10(np.var(signal_to_quantize) / np.var(error))
    snr_estimation = 6 * 10 - 7.2
    print(f'\nUniform signal SNR: {snr_uniform_signal}')
    print(f'Estimation of Uniform signal SNR: {snr_estimation}')
    plot_all_hists_for_signal(signal_to_quantize, quantized_uniform_signal, error, signal_name='Uniform')

    # голос1 (voice.wav)
    _, voice_signal1 = sw.read(cfg.path1)
    # voice_signal1 = voice_signal1 / 2**15
    quantized_voice1_signal, error, signal_to_quantize, _ = get_quantized_signal(voice_signal1)
    snr_voice1_signal = 10 * np.log10(np.var(signal_to_quantize) / np.var(error))
    print(f'\nVoice1 signal SNR: {snr_voice1_signal}')
    plot_all_hists_for_signal(signal_to_quantize, quantized_voice1_signal, error, signal_name='Voice.wav')

    # голос2 (voice2.wav)
    _, voice_signal2 = sw.read(cfg.path2)
    # voice_signal2 = voice_signal2 / 2**15
    quantized_voice2_signal, error, signal_to_quantize, _ = get_quantized_signal(voice_signal2)
    snr_voice2_signal = 10 * np.log10(np.var(signal_to_quantize) / np.var(error))
    print(f'\nVoice2 signal SNR: {snr_voice2_signal}')
    plot_all_hists_for_signal(signal_to_quantize, quantized_voice2_signal, error, signal_name='Voice2.wav')

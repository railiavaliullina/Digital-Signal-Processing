from easydict import EasyDict

cfg = EasyDict()

cfg.zero_freq = 3000
cfg.base_freq = 4000
cfg.one_freq = 5000
cfg.bit_length = 10  # длительность передачи каждого бита
cfg.pause_length = 10  # промежуток между битами

cfg.save_all_plots = False
cfg.path = 'frequency_modulation.wav'

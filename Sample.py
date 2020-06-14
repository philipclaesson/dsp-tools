import utils as u
import metrics as m
import numpy as np

class Sample:
	def __init__(this, target_fs, src = None, data = None):
		assert(src or data), "You must supply either an src string or a data object."
		this.load(src, target_fs)

	def load(this, src, target_fs):
		data, fs = u.read_normalize_sound(src, target_fs)
		this.data = data
		this.fs = fs
		this.duration = 1000 * data.shape[0] / fs

	def set_duration(this, ms):
		"""sets the length in ms, by zero padding or slicing the data"""
		n_samples = int(ms * this.fs / 1000)
		if (this.duration > ms):
			this.data = this.data[0:n_samples]
		else:
			this.data = np.pad(this.data, (0, n_samples - this.data.shape[0]), constant_values = (0)) # zero padding

	def get_STFT(this, window_size = 1024, window_overlap = 512):
		return m.get_STFT(this.data, fs = this.fs, window_size = window_size, window_overlap = window_overlap)

	def get_STFT_db(this, window_size = 1024, window_overlap = 512):
		return m.get_STFT_db(this.data, fs = this.fs, window_size = window_size, window_overlap = window_overlap)

	def play(this, caption = None):
		u.play(this.data, fs = this.fs, caption = caption)

	def show(this, plots = ['soundwave', 'FFT', 'STFT', 'envelope', 'LMS']):
		u.show(this.data, fs = this.fs, plots = plots)

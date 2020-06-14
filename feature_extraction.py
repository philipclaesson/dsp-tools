import numpy as np
from scipy import signal
from scipy.spatial.distance import euclidean
from python_speech_features import mfcc
from dtw import dtw
import utils as u
import metrics as m
import math


def get_features_from_filename(filename, features, bins = None): 
    if (features in ['fft_magnitude', 'fft_log_magnitude', 'fft_db_magnitude']):
        assert bins is not None
        
    x_data = u.read_normalize_sound(filename)
    x_features = get_features_from_data(x_data, features, bins)
    
    return x_features

def get_features_from_data(x_data, fs, features, bins = 5000, downsample_rate = 100): 
    assert features in ['QCT', 'fft_magnitude', 'fft_log_magnitude', 'fft_db_magnitude', 'STFT_db', 'envelope', 'env_gradient', 'mfcc', 'LMS', 'STFT'], print('feature {} not defined. '.format(features))
    if (features == 'fft_magnitude'):
        magnitudes, frequencies = m.get_FMS(x_data, fs, bins = bins)
        return np.reshape(magnitudes, (1, int(bins/2)))
    elif (features == 'fft_log_magnitude'):
        magnitudes, frequencies = m.get_FMS(x_data, fs, bins = bins)
        magnitudes[magnitudes == 0] = 0.000001 # set to small number to avoid floating point err
        log_magnitudes = np.log(magnitudes)
        log_magnitudes[log_magnitudes < 0] = 0 # reset to 0 
        return np.reshape(magnitudes, (1, int(bins/2)))
    elif (features == 'fft_db_magnitude'): 
        magnitudes, frequencies = m.get_FMS(x_data, fs, bins = bins)
        magnitudes[magnitudes == 0] = 0.000001 # set to small number to avoid floating point err
        db_magnitudes = 20 * np.log10(magnitudes/np.max(magnitudes))
        db_magnitudes[db_magnitudes < -30] = -30
        db_magnitudes = db_magnitudes/30 + 1
        return np.reshape(magnitudes, (1, int(bins/2)))
    elif (features == 'envelope'): 
        env = m.get_envelope(x_data, fs)[::downsample_rate] 
        return np.reshape(env, (1, len(env)))
    elif (features == 'env_gradient'):
        env = m.get_envelope(x_data, fs)[::downsample_rate]
        gradient = np.gradient(env)
        return np.reshape(gradient, (1, len(gradient)))
    elif (features == 'mfcc'): 
        return np.reshape(get_mfcc_features(x_data), (1, 39*13))
    elif (features == 'LMS'): # log magnitude spectrogram
        return m.get_LMS(x_data, fs, normalize = True)
    elif (features == 'STFT'): # log magnitude spectrogram
        f, t, Zxx = m.get_STFT(x_data, fs)
        return Zxx
    elif (features == 'STFT_db'): 
        f, t, db_Zxx = m.get_STFT_db(x_data, fs)
        return db_Zxx
    elif (features == 'QCT'): # log magnitude spectrogram
        return m.get_QCT(x_data, fs = 16000)

def get_mfcc_features(data, fs = 88192/2): 
    winlen = 0.1 # 100 ms
    winstep = 0.05 # 50ms
    
    nfft = int(fs * winlen) + 1 # The size of the fft is equivalent to the length of a window.
    
    # Extract feature vectors
    f = mfcc(data, samplerate = fs, numcep = 13, winlen = winlen, winstep = winstep, nfft = nfft)
     
    f = u.normalize(f)
    
    f = np.reshape(f, (f.shape[0]*f.shape[1]))
    
    return f
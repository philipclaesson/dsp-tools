import numpy as np
from scipy import signal
from scipy.spatial.distance import euclidean
from python_speech_features import mfcc
from dtw import dtw
import utils as u
import math
import librosa

def distance(target, candidate, fs_t, fs_c, type = 'FFT'):
    if (type == 'FFT'): return d_FFT(target, candidate, fs_t, fs_c)
    if (type == 'STFT'): return d_STFT(target, candidate, fs_t, fs_c)
    if (type == 'envelope'): return d_envelope(target, candidate, fs_t, fs_c, type = "total")
    if (type == 'envelope-mean'): return d_envelope(target, candidate, fs_t, fs_c, type = "mean")
    if (type == 'envelope-variance'): return d_envelope(target, candidate, fs_t, fs_c, type = "variance")
    if (type == 'DTW'): return d_DTW(target, candidate, fs_t, fs_c)

## FFT
def get_FMS(sound, fs, bins = 5000, return_negative = False, scale = 'linear'):
    """Takes a nparray, returns the Frequency Magnitude Spectrum as magnitudes and frequencies"""
    df = 1/fs # frequency resolution
    DFT = np.fft.fft(sound, bins)
    X_re = np.real(DFT)
    X_im = np.imag(DFT)
    magnitudes = np.sqrt(np.square(X_re) + np.square(X_im))
    frequencies = np.fft.fftfreq(bins, df)
    
    if not return_negative: 
        magnitudes = magnitudes[0:math.floor(bins/2)]
        frequencies = frequencies[0:math.floor(bins/2)]
    
    # Normalize Magnitudes
    if (scale is 'linear'): 
        magnitudes = u.normalize(magnitudes)
        return magnitudes, frequencies

    elif (scale is 'db'): 
        magnitudes[magnitudes == 0] = 0.000001 # set to small number to avoid floating point err
        db_magnitudes = 20 * np.log10(magnitudes/np.max(magnitudes))
        db_magnitudes[db_magnitudes < -30] = -30
        db_magnitudes = db_magnitudes/30 + 1
        return db_magnitudes, frequencies


def d_FFT(target, candidate, fs_t, fs_c): 
    """Takes a target and candidate sound as nparrays, returns the FFT-distance. """
    mag_t, freq_t = get_FMS(target, fs = fs_t)
    mag_c, freq_c = get_FMS(candidate, fs = fs_c)
    
    d_dft = np.sum(np.abs(mag_t - mag_c)) # absolute distance
    
    return d_dft


## STFT
def get_STFT(sound, fs, window_size = 1024, window_overlap = 512, normalize = 'u'):
    """Returns the normalized spectogram of a given sound"""
    f, t, Zxx = signal.stft(sound, fs = fs, nperseg = window_size, noverlap = window_overlap)
    if (normalize == 'u'): 
        Zxx = u.normalize(Zxx)
    return f, t, Zxx

def get_STFT_db(sound, fs, window_size = 1024, window_overlap = 512, normalize = 'u'):
    """Returns the normalized spectogram of a given sound"""
    f, t, Zxx = signal.stft(sound, fs = fs, nperseg = window_size, noverlap = window_overlap)
    db_Zxx = librosa.amplitude_to_db(np.abs(Zxx), ref=np.max)
    if (normalize == 'u'):
        db_Zxx = u.normalize(db_Zxx)
    if (normalize == 'librosa'): 
        db_Zxx = librosa.util.normalize(db_Zxx)
    if (normalize == 'peak-1'): 
        db_Zxx = db_Zxx - np.min(db_Zxx) # bring low to zero
        db_Zxx = db_Zxx/np.max(db_Zxx) # bring high to one

    return f, t, db_Zxx


def d_STFT(target, candidate, fs_t, fs_c): 
    """Takes a target and candidate sound as nparrays, returns the STFT distance. """
    window_size = 1024
    window_overlap = 512
    # n_windows = target.shape[0] / (window_size - window_overlap)
    n_windows = np.ceil((target.shape[0] - window_size) / (window_size - window_overlap))
    # print("window_size: {},  overlap: {}, lemgth: {}, n_windows: {}".format(window_size, window_overlap, target.shape[0], n_windows))
    freq_t, times_t, Z_t = signal.stft(target, fs = fs_t, nperseg = window_size, noverlap = window_overlap)
    freq_c, times_c, Z_c = signal.stft(candidate, fs = fs_c, nperseg = window_size, noverlap = window_overlap)
    
    Z_t = np.abs(Z_t) #  bring to (0,1)
    Z_c = np.abs(Z_c)
    
    Z_t = Z_t - np.min(Z_t) # bring low to zero
    Z_t = Z_t/np.max(Z_t) # bring high to one

    Z_c = Z_c - np.min(Z_c) # bring low to zero
    Z_c = Z_c/np.max(Z_c) # bring high to one

    d_stft = np.sum(np.abs(Z_t - Z_c)) # absolute distance
    return d_stft

def d_STFT_db(target, candidate, fs_t, fs_c): 
    """Takes a target and candidate sound as nparrays, returns the STFT distance. """
    window_size = 1024
    window_overlap = 512
    # n_windows = target.shape[0] / (window_size - window_overlap)
    n_windows = np.ceil((target.shape[0] - window_size) / (window_size - window_overlap))
    # print("window_size: {},  overlap: {}, lemgth: {}, n_windows: {}".format(window_size, window_overlap, target.shape[0], n_windows))
    freq_t, times_t, Z_t = get_STFT_db(target, fs_t, normalize = 'peak-1')
    freq_c, times_c, Z_c = get_STFT_db(candidate, fs_c, normalize = 'peak-1')
    Z_t += 1
    Z_c += 1
    d_stft = np.sum(np.abs(Z_t - Z_c)) # absolute distance
    return d_stft


## Envelope
def get_envelope(sound, fs):
    """ Takes a sound, returns it's amplitude envelope as approximated as the magnitude of the 
    analytic signal obtained through the hilbert transform"""
    
    analytic_signal = signal.hilbert(sound)
    amplitude_envelope = np.abs(analytic_signal)
    
    ## Apply a low pass filter to the envelope
    fc = 30  # Cut-off frequency of the filter
    w = fc / (fs / 2) # Normalize the frequency
    b, a = signal.butter(4, w, 'low')
    filtered_envelope = signal.filtfilt(b, a, amplitude_envelope) # apply filter
    
    filtered_envelope = u.normalize(filtered_envelope)
    
    return filtered_envelope

def d_envelope(target, candidate, fs_t, fs_c, type = "total"):
    env_t = get_envelope(target, fs_t)
    env_c = get_envelope(candidate, fs_c)
    
    diff = env_t - env_c
    
    total = np.sum(np.abs(diff))
    
    mean = (np.sum(np.abs(diff)))/env_t.shape[0]
    
    variance = np.var(diff)
    
    if (type == "total"): return total
    if (type == "mean"): return mean
    if (type == "variance"): return variance

## DTW
def d_DTW(target, candidate, fs_t, fs_c):
    """ Computes the DTW distance using mfcc to extract features from the sound. """

    winlen = 0.1 # 100 ms
    winstep = 0.05 # 50ms
    
    nfft_t = int(fs_t * winlen) # The size of the fft is equivalent to the length of a window.
    nfft_c = int(fs_c * winlen)
    
    # Extract feature vectors
    f_t = mfcc(target, samplerate = fs_t, numcep = 13, winlen = winlen, winstep = winstep, nfft = nfft_t)
    f_c = mfcc(candidate, samplerate = fs_c, numcep = 13, winlen = winlen, winstep = winstep, nfft = nfft_c)
     
    f_t = u.normalize(f_t)
    f_c = u.normalize(f_c)
    
    distance = dtw(f_t, f_c, dist=euclidean)[0]
        
    # distanceFast, path = fastdtw.fastdtw(f_t, f_c, dist=euclidean)
    
    
    return distance #, distanceFast

## log-mel spectrum 

def get_LMS(sound, fs, normalize = True, htk=True):
    mel = librosa.feature.melspectrogram(sound, sr=fs, n_fft=1024, hop_length=512, power=2.0, htk=htk)
    mel[mel == 0] = 1e-9
    mellog = np.log(mel)

    if (normalize == 'peak-1'): 
        mellog = mellog - np.min(mellog) # bring low to zero
        mellog = mellog/np.max(mellog) # bring high to one
        return mellog
    if (normalize == True): 
        melnormalized = librosa.util.normalize(mellog)
        return melnormalized
    else: 
        return mellog # librosa.power_to_db(mel)



def d_LMS(target, candidate, fs_t, fs_c): 
    """Takes a target and candidate sound as nparrays, returns the LMS distance. """
    LMS_t = get_LMS(target, fs_t, 'peak-1')
    LMS_c = get_LMS(candidate, fs_c, 'peak-1')
    
    d_LMS = np.sum(np.abs(LMS_t - LMS_c)) # absolute distance
    
    return d_LMS

def get_QCT(sound, fs, normalize = True): 
    qct = librosa.cqt(sound, sr=fs)
    db_qct = librosa.amplitude_to_db(np.abs(qct), ref=np.max)
    return librosa.util.normalize(db_qct)

def d_QCT(target, candidate, fs_t, fs_c): 
    """Takes a target and candidate sound as nparrays, returns the LMS distance. """
    QCT_t = get_QCT(target, fs_t)
    QCT_c = get_QCT(candidate, fs_c)
    
    d_QCT = np.sum(np.abs(QCT_t - QCT_c)) # absolute distance
    
    return d_QCT

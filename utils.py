import numpy as np
import matplotlib.pyplot as plt
import wavio
import metrics as m
import warnings
import librosa
import librosa.display
from IPython.display import Audio, display


def normalize_signal(signal):
    ## Bring signal to zero mean and max absolute amp = 1 
    np.seterr(divide='raise', invalid='raise')
    if (np.isnan(signal).any() or not np.isfinite(signal).all()): 
        warnings.warn("signal contains {} NaN/inf elements, out of {}.".format(np.count_nonzero(~np.isnan(signal)), signal.shape[0]))
        np.nan_to_num(signal, copy = False)
    
    if (np.max(np.abs(signal)) == 0): 
        warnings.warn("max abs is 0 in data.")
        return signal
    # if (np.sum(np.mean(signal)) != 0):
        # signal = (signal / np.sqrt(np.sum(np.square(signal)) / signal.shape[0])) 
    signal = signal - np.mean(signal)
    signal = signal / np.max(np.abs(signal))
    return signal

def normalize(data, type = "max"):
    ## Takes numpyarray, normalizes by bringing the max value to 1
    np.seterr(divide='raise', invalid='raise')
    if (np.isnan(data).any()): 
        warnings.warn("data contains {} NaN out of {} elements.".format(np.count_nonzero(~np.isnan(data)), data.shape[0]))
        np.nan_to_num(data, copy = False)

    if (np.max(np.abs(data)) == 0): 
        warnings.warn("max abs is 0 in data.", np.abs(data))
        return data

    if (type == "max"): return data/np.max(np.abs(data))

    if (type == "sum"): return data/np.sum(np.abs(data))

def plot_soundwaves(sounds):
    """takes an array of tuples (sound, samplerate, label), plots their soundwaves"""
    for i, sound in enumerate(sounds): 
        signal = sound[0]
        fs = sound[1]
        label = sound[2]
        ax = plt.subplot(int(np.ceil(len(sounds) / 2)), 2, i+1) 
        ax.plot(signal)
        ax.set_title(label)
        ax.set_ylim([-1.1, 1.1])
        ax.set_xlim([0, 80000])
        ax.set_xlabel('time')
        ax.set_ylabel('amplitude')
    plt.show()

def plot_FFT(sound, fs = 41000, bins = 5000, xlim = 'maxfreq', plot_negative = False): 
    magnitudes, frequencies = m.get_FMS(sound, bins = bins, fs = fs, return_negative = plot_negative)
    maxfreq = frequencies[magnitudes.argmax()]
    plt.plot(frequencies, magnitudes)
    if xlim == 'maxfreq': plt.xlim(-10*maxfreq * plot_negative, 10 * maxfreq)
    else: plt.xlim(-1 * xlim * plot_negative, xlim)
    plt.show()    

def read_normalize_sound(filename, fs):
    """ Read a wav file, and resample to resired sample rate. """          
    data, fs = librosa.load(filename, sr = fs) # read and resample

    if np.isnan(data).any():
        warnings.warn("data contains {} NaN out of {} elements.".format(np.count_nonzero(~np.isnan(data)), data.shape))
        np.nan_to_num(data, copy = False)
        
    data = normalize_signal(data)
    
    return data, fs

def patch_as_frac(patch):
    return int(np.multiply(patch, int(0x7fffffff)))

def patch_as_float(patch):
    return np.divide(patch, int(0x7fffffff))

def is_equal(sound1, sound2): 
    return (sound1 == sound2).all()

def play(sound, fs, caption = ""): 
    if (caption): print(caption)
    display(Audio(sound, rate=fs))

def get_top_n_frequencies(sound, fs, n = 1, bins = 5000): 
    magnitudes, frequencies = m.get_FMS(sound, bins = bins, fs = fs, return_negative = False)
    maxfreq = [0] * n
    maxmag = [0] * n

    for i, _ in enumerate(maxfreq): 
        maxfreq[i] = frequencies[magnitudes.argmax()]
        maxmag[i] = magnitudes[magnitudes.argmax()]
        magnitudes[magnitudes.argmax()] = 0

    return maxfreq, maxmag

def show(sound, fs, plots = ['soundwave', 'FFT', 'STFT', 'envelope', 'LMS']): 
        ## PLOTS ## 
        plt.subplots_adjust(wspace = 0.6, hspace =1)
        plt.figure(figsize=(20,10))
        plotcount = 0
        maxfreq = fs/2
        length_seconds = len(sound)/fs
        time = np.arange(0, length_seconds, 1/fs)
        n_rows = np.ceil(len(plots)/2)
        print(n_rows)
        if ('soundwave' in plots): 
            plotcount += 1
            ax = plt.subplot(n_rows, 2, plotcount)
            ax.plot(time, sound)
            ax.set_ylim([-1.1, 1.1])
            ax.set_xlim([0, length_seconds])
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Amplitude (normalized)')


        if ('envelope' in plots): 
            plotcount += 1
            ax = plt.subplot(n_rows, 2, plotcount) 
            envelope = m.get_envelope(sound, fs = fs)
            ax.plot(time, envelope)
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Amplitude (normalized)')


        if ('STFT' in plots): 
            plotcount += 1
            ax = plt.subplot(n_rows, 2, plotcount)
            f, t, Zxx = m.get_STFT(sound, fs = fs)
            c = ax.pcolormesh(t, f, np.abs(Zxx))
            plt.colorbar(c, label = 'Amplitude (normalized)') # .set_label('Magnitude (normalized)')
            ax.set_ylim([0, maxfreq])
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Frequency [Hz]')


        if ('FFT' in plots): 
            plotcount += 1
            ax = plt.subplot(n_rows, 2, plotcount)
            # ax.set_title("Target FFT")
            magnitudes, frequencies = m.get_FMS(sound, fs = fs, bins = 5000, return_negative = False, scale = 'db')
            plt.plot(frequencies, magnitudes)
            plt.xlim(-100, maxfreq)
            ax.set_ylim(0, 1)
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel('Magnitude (dB)')

        if ('LMS' in plots): 
            plotcount += 1
            ax = plt.subplot(n_rows, 2, plotcount)
            db_S = m.get_LMS(sound, fs)
            # time = np.arange(0, 4, 4/db_S.shape[1])
            # freqs = librosa.core.mel_frequencies(fmin=0.0, fmax=fs/2, n_mels=128)
            librosa.display.specshow(db_S, sr = fs, y_axis='mel', fmax=8000, x_axis='time')
            # print(freqs)
            # c = ax.pcolormesh(time, freqs, db_S)
            # plt.colorbar(c) # .set_label('Magnitude (normalized)')
            # librosa.display.specshow(db_S)
            plt.colorbar(format='%+2.0f dB', label = 'Amplitude [dB]')
            ax.set_ylim(0, 8100)
            ax.set_xlabel('time [s]')
            ax.set_ylabel('frequency [Hz]')

        
        plt.show()


def plot_LMS(sound, fs, normalize = True): 
    db_S = m.get_LMS(sound, fs, normalize)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(db_S)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    plt.show()

def resample(y, orig_sr, target_sr, res_type='kaiser_best', fix=True, scale=False, **kwargs):
    """From librosa: Resample a time series from orig_sr to target_sr
    Parameters
    ----------
    y : np.ndarray [shape=(n,) or shape=(2, n)]
        audio time series.  Can be mono or stereo.
    orig_sr : number > 0 [scalar]
        original sampling rate of `y`
    target_sr : number > 0 [scalar]
        target sampling rate
    res_type : str
        resample type (see note)
        .. note::
            By default, this uses `resampy`'s high-quality mode ('kaiser_best').
            To use a faster method, set `res_type='kaiser_fast'`.
            To use `scipy.signal.resample`, set `res_type='fft'` or `res_type='scipy'`.
            To use `scipy.signal.resample_poly`, set `res_type='polyphase'`.
        .. note::
            When using `res_type='polyphase'`, only integer sampling rates are
            supported.
    fix : bool
        adjust the length of the resampled signal to be of size exactly
        `ceil(target_sr * len(y) / orig_sr)`
    scale : bool
        Scale the resampled signal so that `y` and `y_hat` have approximately
        equal total energy.
    kwargs : additional keyword arguments
        If `fix==True`, additional keyword arguments to pass to
        `librosa.util.fix_length`.
    Returns
    -------
    y_hat : np.ndarray [shape=(n * target_sr / orig_sr,)]
        `y` resampled from `orig_sr` to `target_sr`
    Raises
    ------
    ParameterError
        If `res_type='polyphase'` and `orig_sr` or `target_sr` are not both
        integer-valued.
    See Also
    --------
    librosa.util.fix_length
    scipy.signal.resample
    resampy.resample
    Notes
    -----
    This function caches at level 20.
    Examples
    --------
    Downsample from 22 KHz to 8 KHz
    >>> y, sr = librosa.load(librosa.util.example_audio_file(), sr=22050)
    >>> y_8k = librosa.resample(y, sr, 8000)
    >>> y.shape, y_8k.shape
    ((1355168,), (491671,))
    """

    # First, validate the audio buffer
    librosa.util.valid_audio(y, mono=False)

    if orig_sr == target_sr:
        return y

    ratio = float(target_sr) / orig_sr

    n_samples = int(np.ceil(y.shape[-1] * ratio))

    if res_type in ('scipy', 'fft'):
        y_hat = scipy.signal.resample(y, n_samples, axis=-1)
    elif res_type == 'polyphase':
        if int(orig_sr) != orig_sr or int(target_sr) != target_sr:
            raise ParameterError('polyphase resampling is only supported for integer-valued sampling rates.')

        # For polyphase resampling, we need up- and down-sampling ratios
        # We can get those from the greatest common divisor of the rates
        # as long as the rates are integrable
        orig_sr = int(orig_sr)
        target_sr = int(target_sr)
        gcd = np.gcd(orig_sr, target_sr)
        y_hat = scipy.signal.resample_poly(y, target_sr // gcd, orig_sr // gcd, axis=-1)
    else:
        y_hat = resampy.resample(y, orig_sr, target_sr, filter=res_type, axis=-1)

    if fix:
        y_hat = librosa.util.fix_length(y_hat, n_samples, **kwargs)

    if scale:
        y_hat /= np.sqrt(ratio)

    return np.ascontiguousarray(y_hat, dtype=y.dtype)


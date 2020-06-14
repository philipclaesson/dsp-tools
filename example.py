from Sample import Sample

s = Sample(src = 'samples/CrashCymbal.wav', target_fs = 48000)
s.set_duration(2000)
s.get_STFT_db()
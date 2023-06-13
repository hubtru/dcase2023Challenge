import librosa
import numpy as np
from scipy.ndimage import zoom
from scipy import interpolate

path = r'C:\Users\HABELS.COMPUTACENTER\Downloads\dcase_training_data\bearing\test\section_00_source_test_anomaly_0000_vel_16_loc_D.wav'

audio, sr = librosa.load(path, sr=None)
print(audio.shape)
print(sr)

y, sr = librosa.load(path)  # Replace 'audio.wav' with your audio file
n_mfcc = 128  # Desired number of MFCC coefficients
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

# Set the desired output size
output_size = (20, 431)

# Get the current MFCC size
current_size = mfcc.shape

# Compute the scaling factor for the time axis
scale_factor_time = current_size[1] / output_size[1]

# Create the rescaling function for the time axis using cubic spline interpolation
rescale_time = interpolate.interp1d(
    np.arange(current_size[1]), mfcc, axis=1, kind='cubic'
)

# Perform rescaling
rescaled_mfcc = rescale_time(np.arange(output_size[1]) * scale_factor_time)


# Adjust the output size to match the desired shape
#rescaled_mfcc = rescaled_mfcc[:, :output_size[1]]

print(rescaled_mfcc.shape)  # Output: (20, 100)
print(mfcc[0])
print(rescaled_mfcc[0])
print(np.allclose(rescaled_mfcc, mfcc))  # Output: True



stft = librosa.stft(audio)
print(stft.shape)

# Set the desired output size
output_size = (20, 100)

# Get the current STFT size
current_size = stft.shape

# Compute the scaling factors for the frequency and time axes
scale_factor_freq = output_size[0] / current_size[0]
scale_factor_time = output_size[1] / current_size[1]

# Perform rescaling
rescaled_stft = zoom(stft, (scale_factor_freq, scale_factor_time))

print(rescaled_stft.shape)  # Output: (20, 100)

stft = librosa.stft(y=audio, )
print(stft.shape)


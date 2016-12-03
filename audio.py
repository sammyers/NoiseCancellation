import wave
import numpy as np
from scipy.io import wavfile

class NoiseCanceller(object):

    def __init__(self, step_size=0.0003, num_samples=100):
        self.step_size = step_size
        self.num_samples = num_samples

    def load_audio(self, filepath: str) -> (int, np.array):
        """
        Return a numpy array for a given audio file with its sample rate.
        """
        sample_rate, audio_array = wavfile.read(filepath)
        return sample_rate, audio_array

    def subtract_signal(self, input_signal, original_signal):
        """
        Return the difference of two audio signals.
        """
        original_signal = np.resize(original_signal, input_signal.shape)

        return input_signal - original_signal

    def filter_signal(self, error_signal, noise_signal) -> np.array:
        """
        Process the error signal based on an adaptive filter.
        """
        filter_output = np.empty_like(error_signal)
        coeffs = np.zeros(self.num_samples)
        for n in range(self.num_samples, error_signal.size):
            moving_signal = noise_signal[n:n-self.num_samples:-1]
            coeffs = coeffs - self.step_size * error_signal[n] * moving_signal
            filter_output[n] = np.dot(coeffs, moving_signal)

        return filter_output

    def cancel_noise(self, input_signal, original_signal, noise_signal):
        """
        Hopefully cancel some noise.
        """
        error_signal = self.subtract_signal(input_signal, original_signal)
        filtered_signal = self.filter_signal(error_signal, noise_signal)
        new_signal = input_signal - filtered_signal
        return error_signal

if __name__ == '__main__':
    nc = NoiseCanceller()
    sr, haddaway = nc.load_audio('/Users/sam/Desktop/haddaway.wav')
    sr2, headphone = nc.load_audio('/Users/sam/git/NoiseCancellation/new_left2.wav')
    sr3, noise = nc.load_audio('/Users/sam/git/NoiseCancellation/new_right2.wav')
    haddaway = haddaway[:, 0].flatten()
    headphone = headphone[:, 0].flatten()
    noise = noise[:, 1].flatten()
    cancelled = nc.cancel_noise(headphone, haddaway, noise)
    wavfile.write('/Users/sam/git/NoiseCancellation/cancelled.wav', sr3, cancelled)

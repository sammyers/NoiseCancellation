import wave
import numpy as np
from scipy.io import wavfile

class NoiseCanceller(object):

    def __init__(self, step_size=0.003, num_samples=100):
        self.step_size = step_size
        self.num_samples = num_samples

    def load_audio(filepath: str) -> (int, np.array):
        """
        Return a numpy array for a given audio file with its sample rate.
        """
        with wave.open(filepath) as audio_file:
            sample_rate, audio_array = wavfile.read(audio_file)
        return sample_rate, audio_array

    def subtract_signal(signal1, signal2):
        """
        Return the difference of two audio signals.
        """
        if signal1.size > signal2.size:
            signal2.resize(signal1.size)
        elif signal1.size < signal2.size:
            signal1.resize(signal2.size)

        return signal1 - signal2

    def compute_filter(prev_value: float, error_value: float, noise_signal: np.array) -> float:
        """
        Calculate the next value of the filter signal.
        """
        return prev_value + (self.step_size * error_value * noise_signal)

    def filter_signal(error_signal, noise_signal) -> np.array:
        """
        Process the error signal based on an adaptive filter.
        """
        filter_output = np.empty_like(error_signal)
        coeffs = np.zeros(self.num_samples)

        for n in range(error_signal.size):
            moving_signal = noise_signal[n:n-self.num_samples+1:-1]
            coeffs = coeffs + self.step_size * error_signal[n] * moving_signal
            filter_output[n] = np.dot(coeffs, moving_signal)

        return filter_output
        
    def cancel_noise(input_signal, original_signal, noise_signal):
        """
        Hopefully cancel some noise.
        """
        error_signal = subtract_signal(input_signal, original_signal)
        filtered_signal = filter_signal(error_signal, noise_signal)
        new_signal = input_signal - filtered_signal
        return new_signal

if __name__ == '__main__':
    pass

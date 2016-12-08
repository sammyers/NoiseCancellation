import wave
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import sounddevice as sd

class NoiseCanceller(object):

    def __init__(self, step_size=0.1, num_samples=500):
        self.step_size = step_size
        self.num_samples = num_samples

    def load_audio(self, filepath: str) -> (int, np.array):
        """
        Return a numpy array for a given audio file with its sample rate.
        """
        sample_rate, audio_array = wavfile.read(filepath)
        return sample_rate, audio_array

    def cancel_noise(self, input_signal, noise_signal):
        """
        Hopefully cancel some noise.
        """
        error_signal = np.zeros(input_signal.size)
        filter_output = np.zeros(input_signal.size)
        coeffs = np.zeros(self.num_samples)
        for n in range(self.num_samples, input_signal.size):
            moving_noise = noise_signal[n:n-self.num_samples:-1]
            filter_output[n] = np.dot(coeffs, moving_noise)
            error_signal[n] = input_signal[n] - filter_output[n]
            coeffs = coeffs + (self.step_size * error_signal[n] * moving_noise / (np.linalg.norm(moving_noise) ** 2))
        return error_signal

if __name__ == '__main__':
    np.seterr(all='raise')
    nc = NoiseCanceller()
    sr2, headphone = nc.load_audio('/Users/sam/git/NoiseCancellation/left_noise.wav')
    sr3, noise = nc.load_audio('/Users/sam/git/NoiseCancellation/right_noise.wav')
    headphone = headphone[:, 0].flatten()[20000:200000]
    noise = noise[:, 1].flatten()[20000:200000]
    error = nc.cancel_noise(headphone, noise)
    wavfile.write('/Users/sam/git/NoiseCancellation/left_processed.wav', sr3, headphone)
    wavfile.write('/Users/sam/git/NoiseCancellation/right_processed.wav', sr3, noise)
    wavfile.write('/Users/sam/git/NoiseCancellation/cancelled.wav', sr3, error)
    plt.figure(1)

    ax2 = plt.subplot(311)
    ax2.set_title('Internal microphone')
    ax2.plot(headphone[::4])
    ax2.get_xaxis().set_visible(False)
    ax2.set_ylim([-20000, 20000])

    ax3 = plt.subplot(312)
    ax3.set_title('External noise')
    ax3.plot(noise[::4])
    ax3.get_xaxis().set_visible(False)
    ax3.set_ylim([-20000, 20000])

    ax4 = plt.subplot(313)
    ax4.set_title('Error signal')
    ax4.plot(error[::4])
    ax4.get_xaxis().set_visible(False)
    ax4.set_ylim([-20000, 20000])

    plt.show()

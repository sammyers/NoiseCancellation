import wave
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

class NoiseCanceller(object):

    def __init__(self, step_size=0.03, num_samples=100):
        self.step_size = step_size
        self.num_samples = num_samples

    def load_audio(self, filepath: str) -> (int, np.array):
        """
        Return a numpy array for a given audio file with its sample rate.
        """
        sample_rate, audio_array = wavfile.read(filepath)
        return sample_rate, audio_array

    def cancel_noise(self, input_signal, original_signal, noise_signal):
        """
        Hopefully cancel some noise.
        """
        error_signal = np.zeros(input_signal.size)
        lms_song = np.zeros(input_signal.size)
        lms_noise = np.zeros(input_signal.size)
        noise_cancelled = np.zeros(input_signal.size)
        song_coeffs = np.zeros(self.num_samples)
        noise_coeffs = np.zeros(self.num_samples)
        for n in range(self.num_samples, input_signal.size):
            moving_noise = noise_signal[n:n-self.num_samples:-1]
            moving_song = original_signal[n:n-self.num_samples:-1]
            lms_song[n] = np.dot(song_coeffs, moving_song)
            lms_noise[n] = np.dot(noise_coeffs, moving_noise)
            error_signal[n] = input_signal[n] - lms_noise[n] - lms_song[n]
            song_coeffs = song_coeffs + (self.step_size * error_signal[n]) * moving_song
            noise_coeffs = noise_coeffs + (self.step_size * error_signal[n]) * moving_noise
            noise_cancelled[n] = input_signal[n] - np.dot(noise_coeffs, moving_noise)
            # if n == 44000:
            #     print(n)
            #     # print(e)
            #     print(coeffs)
            #     print(error_signal[n])
            #     print(moving_signal)
            #     break
        return error_signal, noise_cancelled

if __name__ == '__main__':
    np.seterr(all='raise')
    nc = NoiseCanceller()
    sr, haddaway = nc.load_audio('/Users/sam/Desktop/haddaway.wav')
    sr2, headphone = nc.load_audio('/Users/sam/git/NoiseCancellation/new_left3.wav')
    sr3, noise = nc.load_audio('/Users/sam/git/NoiseCancellation/new_right3.wav')
    haddaway = haddaway[:, 0].flatten()
    headphone = headphone[:, 0].flatten()
    noise = noise[:, 1].flatten()
    error, cancelled = nc.cancel_noise(headphone, haddaway, noise)
    wavfile.write('/Users/sam/git/NoiseCancellation/cancelled.wav', sr3, cancelled[100:])
    plt.figure(1)

    ax1 = plt.subplot(511)
    ax1.set_title('Original music')
    ax1.plot(np.resize(haddaway, headphone.shape)[::4])
    ax1.get_xaxis().set_visible(False)

    ax2 = plt.subplot(512)
    ax2.set_title('Internal microphone')
    ax2.plot(headphone[::4])
    ax2.get_xaxis().set_visible(False)

    ax3 = plt.subplot(513)
    ax3.set_title('External noise')
    ax3.plot(noise[::4])
    ax3.get_xaxis().set_visible(False)

    ax4 = plt.subplot(514)
    ax4.set_title('Error signal')
    ax4.plot(error[::4])
    # ax4.get_xaxis().set_visible(False)

    ax5 = plt.subplot(515)
    ax5.set_title('Filtered signal')
    ax5.plot(cancelled[::4])
    ax5.get_xaxis().set_visible(False)

    plt.show()

from importlib import import_module
from scipy.signal import find_peaks, stft, welch
from scipy.fft import rfft, rfftfreq
from flexible_rPPG.utils import *


def get_bpm(signal, fps, type, remove_outlier, params):

    estimator_module = import_module('flexible_rPPG.hr_estimator')
    estimator_method = getattr(estimator_module, type)

    if (type == 'stft_estimator') and ('bpm_type' in params):
        params_copy = params.copy()
        del params_copy['bpm_type']  # Delete bpm type since stft will give continuous hr values
        hr = estimator_method(signal, fps, remove_outlier, **params_copy)
    else:
        hr = estimator_method(signal, fps, remove_outlier, **params)

    hr = np.mean(hr)
    return hr


def stft_estimator(signal, fps, remove_outlier, signal_length, increment, mask):

    if signal.ndim == 1:
        noverlap = fps * (signal_length - increment)
        nperseg = fps * signal_length  # Length of fourier window

        freqs, times, Zxx = stft(x=signal, fs=fps, nperseg=nperseg, noverlap=noverlap)  # Perform STFT
        magnitude_Zxx = np.abs(Zxx).T  # Calculate the magnitude of Zxx
        frequencies = np.repeat(freqs[np.newaxis, :], magnitude_Zxx.shape[0], axis=0)

    elif signal.ndim == 2:
        frequencies = []
        magnitude_Zxx = []

        for window in signal:
            freqs = rfftfreq(len(window), d=1 / fps)
            magnitudes = np.abs(rfft(window)) ** 2

            frequencies.append(freqs)
            magnitude_Zxx.append(magnitudes)

    else:
        raise ValueError("Signal must be either 1D or 2D array")

    frequencies = np.array(frequencies)
    magnitude = np.array(magnitude_Zxx)
    hr = []
    prev_hr = None

    if remove_outlier:
        for freqs, power_spectrum in zip(frequencies, magnitude):
            # Find the maximum peak between 0.75 Hz and 4 Hz
            freq_mask = (freqs >= mask[0]) & (freqs <= mask[1])
            filtered_power_spectrum = power_spectrum[freq_mask]
            filtered_freqs = freqs[freq_mask]

            peaks, _ = find_peaks(filtered_power_spectrum)  # index of the peaks

            if len(peaks) == 0:
                if prev_hr is not None:
                    hr.append(prev_hr)
                continue

            peak_freqs = filtered_freqs[peaks]  # corresponding peaks frequencies
            peak_powers = filtered_power_spectrum[peaks]  # corresponding peaks powers
            max_peak_frequency = peak_freqs[np.argmax(peak_powers)]
            hr_estimated = max_peak_frequency * 60

            # For the first previous HR value
            if prev_hr is None:
                prev_hr = hr_estimated

            while abs(prev_hr - hr_estimated) >= 12:
                max_peak_mask = (peak_freqs == max_peak_frequency)
                peak_freqs = peak_freqs[~max_peak_mask]
                peak_powers = peak_powers[~max_peak_mask]

                if len(peak_freqs) == 0:
                    hr_estimated = prev_hr
                    break

                max_peak_frequency = peak_freqs[np.argmax(peak_powers)]
                hr_estimated = max_peak_frequency * 60

            prev_hr = hr_estimated
            hr.append(hr_estimated)

    else:
        for freqs, power_spectrum in zip(frequencies, magnitude):
            # Find the maximum peak between 0.75 Hz and 4 Hz
            freq_mask = (freqs >= mask[0]) & (freqs <= mask[1])
            filtered_power_spectrum = power_spectrum[freq_mask]
            filtered_freqs = freqs[freq_mask]

            peaks, _ = find_peaks(filtered_power_spectrum)  # index of the peaks

            if len(peaks) == 0:
                if prev_hr is not None:
                    hr.append(prev_hr)
                continue

            peak_freqs = filtered_freqs[peaks]
            peak_powers = filtered_power_spectrum[peaks]
            max_peak_frequency = peak_freqs[np.argmax(peak_powers)]
            hr_estimated = max_peak_frequency * 60

            prev_hr = hr_estimated
            hr.append(hr_estimated)

    return hr


def fft_estimator(signal, fps, remove_outlier, bpm_type, signal_length, increment, mask):

    if bpm_type == 'average':
        # Compute the positive frequencies and the corresponding power spectrum
        freqs = rfftfreq(len(signal), d=1 / fps)
        power_spectrum = np.abs(rfft(signal)) ** 2

        # Find the maximum peak between 0.75 Hz and 4 Hz
        freq_mask = (freqs >= mask[0]) & (freqs <= mask[1])
        filtered_power_spectrum = power_spectrum[freq_mask]
        filtered_freqs = freqs[freq_mask]

        peaks, _ = find_peaks(filtered_power_spectrum)  # index of the peaks
        peak_freqs = filtered_freqs[peaks]  # corresponding peaks frequencies
        peak_powers = filtered_power_spectrum[peaks]  # corresponding peaks powers

        hr = peak_freqs[np.argmax(peak_powers)] * 60

    elif bpm_type == 'continuous':
        if signal.ndim == 2:
            windowed_sig = signal
        else:
            windowed_sig = moving_window(sig=signal, fps=fps, window_size=signal_length, increment=increment)

        frequencies = []
        magnitude = []
        hr = []
        prev_hr = None

        for each_sig in windowed_sig:
            # Compute the positive frequencies and the corresponding power spectrum
            freqs = rfftfreq(len(each_sig), d=1 / fps)
            power_spectrum = np.abs(rfft(each_sig)) ** 2

            frequencies.append(freqs)
            magnitude.append(power_spectrum)

        frequencies = np.array(frequencies)
        magnitude = np.array(magnitude)

        if remove_outlier:
            for freqs, power_spectrum in zip(frequencies, magnitude):
                # Find the maximum peak between 0.75 Hz and 4 Hz
                freq_mask = (freqs >= mask[0]) & (freqs <= mask[1])
                filtered_power_spectrum = power_spectrum[freq_mask]
                filtered_freqs = freqs[freq_mask]

                peaks, _ = find_peaks(filtered_power_spectrum)  # index of the peaks

                if len(peaks) == 0:
                    if prev_hr is not None:
                        hr.append(prev_hr)
                    continue

                peak_freqs = filtered_freqs[peaks]  # corresponding peaks frequencies
                peak_powers = filtered_power_spectrum[peaks]  # corresponding peaks powers
                max_peak_frequency = peak_freqs[np.argmax(peak_powers)]
                hr_estimated = max_peak_frequency * 60

                # For the first previous HR value
                if prev_hr is None:
                    prev_hr = hr_estimated

                while abs(prev_hr - hr_estimated) >= 12:
                    max_peak_mask = (peak_freqs == max_peak_frequency)
                    peak_freqs = peak_freqs[~max_peak_mask]
                    peak_powers = peak_powers[~max_peak_mask]

                    if len(peak_freqs) == 0:
                        hr_estimated = prev_hr
                        break

                    max_peak_frequency = peak_freqs[np.argmax(peak_powers)]
                    hr_estimated = max_peak_frequency * 60

                prev_hr = hr_estimated
                hr.append(hr_estimated)
        else:
            for freqs, power_spectrum in zip(frequencies, magnitude):
                # Find the maximum peak between 0.75 Hz and 4 Hz
                freq_mask = (freqs >= mask[0]) & (freqs <= mask[1])
                filtered_power_spectrum = power_spectrum[freq_mask]
                filtered_freqs = freqs[freq_mask]

                peaks, _ = find_peaks(filtered_power_spectrum)  # index of the peaks

                if len(peaks) == 0:
                    if prev_hr is not None:
                        hr.append(prev_hr)
                    continue

                peak_freqs = filtered_freqs[peaks]  # corresponding peaks frequencies
                peak_powers = filtered_power_spectrum[peaks]  # corresponding peaks powers
                hr_estimated = peak_freqs[np.argmax(peak_powers)] * 60

                prev_hr = hr_estimated
                hr.append(hr_estimated)

    return hr


def welch_estimator(signal, fps, remove_outlier, bpm_type, signal_length, increment, mask):

    if bpm_type == 'average':
        frequencies, psd = welch(signal, fs=fps, nperseg=len(signal), nfft=8192)

        first_index = np.where(frequencies > mask[0])[0][0]
        last_index = np.where(frequencies < mask[1])[0][-1]
        range_of_interest = range(first_index, last_index + 1, 1)
        max_idx = np.argmax(psd[range_of_interest])
        f_max = frequencies[range_of_interest[max_idx]]
        hr = f_max * 60.0

    elif bpm_type == 'continuous':
        if signal.ndim == 2:
            windowed_sig = signal
        else:
            windowed_sig = moving_window(sig=signal, fps=fps, window_size=signal_length, increment=increment)

        frequencies = []
        magnitude = []
        hr = []
        prev_hr = None

        for each_sig in windowed_sig:
            freqs, psd = welch(each_sig, fs=fps, nperseg=len(each_sig), nfft=8192)

            frequencies.append(freqs)
            magnitude.append(psd)

        frequencies = np.array(frequencies)
        magnitude = np.array(magnitude)

        if remove_outlier:
            for freqs, power_spectrum in zip(frequencies, magnitude):

                if len(freqs) == 0 or len(power_spectrum) == 0:
                    if prev_hr is not None:
                        hr.append(prev_hr)
                    continue

                first_index = np.where(freqs > mask[0])[0][0]
                last_index = np.where(freqs < mask[1])[0][-1]
                range_of_interest = range(first_index, last_index + 1, 1)
                max_idx = np.argmax(power_spectrum[range_of_interest])
                f_max = freqs[range_of_interest[max_idx]]
                hr_estimated = f_max * 60.0
                hr.append(hr_estimated)

                # For the first previous HR value
                if prev_hr is None:
                    prev_hr = hr_estimated

                while abs(prev_hr - hr_estimated) >= 12:
                    index_mask = freqs != f_max
                    freqs = freqs[index_mask]
                    power_spectrum = power_spectrum[index_mask]
                    first = np.where(freqs > mask[0])[0]
                    last = np.where(freqs < mask[1])[0]

                    # If no more frequencies left in the range, set the estimated heart rate to the previous value and break
                    if first.size == 0 or last.size == 0:
                        hr_estimated = prev_hr
                        break

                    first_index = first[0]
                    last_index = last[-1]
                    range_of_interest = range(first_index, last_index + 1, 1)
                    max_idx = np.argmax(power_spectrum[range_of_interest])
                    f_max = freqs[range_of_interest[max_idx]]
                    hr_estimated = f_max * 60.0

                prev_hr = hr_estimated
                hr.append(hr_estimated)

        else:
            for freqs, power_spectrum in zip(frequencies, magnitude):

                if len(freqs) == 0 or len(power_spectrum) == 0:
                    if prev_hr is not None:
                        hr.append(prev_hr)
                    continue

                first_index = np.where(freqs > mask[0])[0][0]
                last_index = np.where(freqs < mask[1])[0][-1]
                range_of_interest = range(first_index, last_index + 1, 1)
                max_idx = np.argmax(power_spectrum[range_of_interest])
                f_max = freqs[range_of_interest[max_idx]]
                hr_estimated = f_max * 60.0

                prev_hr = hr_estimated
                hr.append(hr_estimated)

    return hr


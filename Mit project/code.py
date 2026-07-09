This folder contains project code files
Audio Sensor code:
import numpy as np
import sounddevice as sd
import time
from collections import deque

SAMPLE_RATE = 44100
CHUNK = 1024
CONSECUTIVE_FRAMES_REQUIRED = 3
NOISE_FLOOR_WINDOW = 100          # frames used for rolling adaptive baseline (~2.3s worth)
ONSET_RATIO_REQUIRED = 3.0        # frame must be this many x the rolling noise floor
FLATNESS_MIN_FOR_COUGH = 0.15     # coughs are noisy/broadband -> higher spectral flatness
PRINT_INTERVAL_SECONDS = 1.0      # how often you SEE output, not how often it analyzes


def compute_features(audio_chunk):
    """Extract volume, dominant frequency, spectral flatness, and zero-crossing rate."""
    samples = (audio_chunk[:, 0] if audio_chunk.ndim > 1 else audio_chunk).astype(np.float64)

    rms = np.sqrt(np.mean(samples ** 2))
    vol_peak = rms * 1000

    windowed = samples * np.hanning(len(samples))
    fft_vals = np.fft.rfft(windowed)
    fft_freqs = np.fft.rfftfreq(len(windowed), d=1.0 / SAMPLE_RATE)
    magnitude = np.abs(fft_vals) + 1e-12  # avoid log(0)

    freq = 0.0 if magnitude.max() < 1e-6 else fft_freqs[np.argmax(magnitude)]

    # Spectral flatness: geometric_mean / arithmetic_mean of the spectrum.
    # Near 1.0 = noise-like/broadband (cough, hiss). Near 0 = tonal/narrowband (hum, whistle, steady AC).
    log_mag = np.log(magnitude)
    geo_mean = np.exp(np.mean(log_mag))
    arith_mean = np.mean(magnitude)
    flatness = geo_mean / arith_mean

    # Zero-crossing rate: how often the signal flips sign per sample.
    # High for noisy/turbulent sounds like coughs, breath, fricatives; lower for smooth tones/hums.
    zcr = np.mean(np.abs(np.diff(np.sign(samples)))) / 2

    return vol_peak, freq, flatness, zcr


def run_audio_sensor():
    print("Mic Active. Initializing Audio Triage Stream Monitoring Engine...")
    print("Press Ctrl + C in the terminal window to halt.")

    noise_floor_history = deque(maxlen=NOISE_FLOOR_WINDOW)
    consecutive_hits = 0
    recent_freqs = deque(maxlen=CONSECUTIVE_FRAMES_REQUIRED)
    last_print_time = 0.0

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=CHUNK) as stream:
            print("Calibrating ambient noise for 3s... stay quiet.")
            for _ in range(int(3 * SAMPLE_RATE / CHUNK)):
                audio_chunk, _ = stream.read(CHUNK)
                vol_peak, _, _, _ = compute_features(audio_chunk)
                noise_floor_history.append(vol_peak)
            print(f"Calibration done. Initial noise floor: {np.mean(noise_floor_history):.1f}\n")

            while True:
                audio_chunk, _ = stream.read(CHUNK)
                vol_peak, freq, flatness, zcr = compute_features(audio_chunk)

                # Rolling adaptive noise floor — median is robust to occasional loud spikes
                noise_floor = np.median(noise_floor_history)
                onset_ratio = vol_peak / max(noise_floor, 1e-6)

                is_sudden_and_loud = onset_ratio > ONSET_RATIO_REQUIRED
                is_broadband = flatness > FLATNESS_MIN_FOR_COUGH

                recent_freqs.append(freq)

                if is_sudden_and_loud and is_broadband:
                    consecutive_hits += 1
                else:
                    consecutive_hits = 0
                    # Only let quiet/steady frames update the noise floor
                    noise_floor_history.append(vol_peak)

                status = "Normal Ambient Sound"
                if consecutive_hits >= CONSECUTIVE_FRAMES_REQUIRED:
                    avg_freq = np.mean(recent_freqs)
                    if 600.0 <= avg_freq <= 1000.0:
                        status = "Wheezing / Stridor Detected"
                    elif 100.0 <= avg_freq <= 800.0:
                        status = "Severe Coughing Fit Detected"
                    else:
                        status = "Sudden Loud Noise (Unclassified)"
                elif onset_ratio > ONSET_RATIO_REQUIRED and not is_broadband:
                    status = "Steady Background Noise / Continuous Audio"

                # A detected event always prints immediately, regardless of pacing,
                # so you never miss the moment it actually fires.
                now = time.time()
                should_print = (now - last_print_time >= PRINT_INTERVAL_SECONDS) or (status != "Normal Ambient Sound")

                if should_print:
                    print(f"Mic Active | Vol: {vol_peak:.1f} | Floor: {noise_floor:.1f} | "
                          f"Freq: {freq:.1f} Hz | Hits: {consecutive_hits} | Status: {status}")
                    last_print_time = now

    except KeyboardInterrupt:
        print("\nAudio monitoring stream stopped cleanly.")


if __name__ == "__main__":
    run_audio_sensor()
Video Sensor code :

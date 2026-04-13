import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# ---------------- LOAD AUDIO ---------------- #
def load_audio():
    Tk().withdraw()  # hide tkinter window
    filename = askopenfilename(title="Select a WAV file")

    signal, sr = sf.read(filename)

    # Convert stereo to mono
    if signal.ndim == 2:
        signal = np.mean(signal, axis=1)

    # Limit to first 3 seconds
    signal = signal[:sr * 3]

    t = np.linspace(0, len(signal)/sr, len(signal))

    print(f"\nLoaded File: {filename}")
    print(f"Sample Rate : {sr} Hz")
    print(f"Duration    : {len(signal)/sr:.2f} sec")

    return t, signal, sr


# ---------------- FFT SPECTRUM ---------------- #
def get_spectrum(signal, sr):
    fft_vals = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(len(signal), d=1/sr)
    pos = fft_freq >= 0
    return fft_freq[pos], np.abs(fft_vals[pos])


# ---------------- FILTERS ---------------- #
def low_pass(signal, sr):
    fft_vals = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(len(signal), d=1/sr)
    fft_vals[np.abs(fft_freq) > 1000] = 0
    return np.fft.ifft(fft_vals).real


def high_pass(signal, sr):
    fft_vals = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(len(signal), d=1/sr)
    fft_vals[np.abs(fft_freq) < 1000] = 0
    return np.fft.ifft(fft_vals).real


def band_pass(signal, sr):
    fft_vals = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(len(signal), d=1/sr)
    mask = (np.abs(fft_freq) < 500) | (np.abs(fft_freq) > 3000)
    fft_vals[mask] = 0
    return np.fft.ifft(fft_vals).real


# ---------------- SAVE AUDIO ---------------- #
def save_audio(filtered, sr, name):
    filtered = filtered / np.max(np.abs(filtered))
    filename = f"filtered_{name}.wav"
    sf.write(filename, filtered, sr)
    print(f"Saved: {filename}")


# ---------------- PLOT ---------------- #
def plot(t, original, filtered, sr, name):
    f1, a1 = get_spectrum(original, sr)
    f2, a2 = get_spectrum(filtered, sr)

    fig, ax = plt.subplots(2, 2, figsize=(12, 6))
    fig.suptitle(name)

    # Original Time
    ax[0, 0].plot(t, original, linewidth=0.5)
    ax[0, 0].set_title("Original - Time Domain")

    # Original FFT
    ax[0, 1].plot(f1, a1, linewidth=0.5)
    ax[0, 1].set_title("Original - Frequency Spectrum")

    # Filtered Time
    ax[1, 0].plot(t, filtered, linewidth=0.5)
    ax[1, 0].set_title("Filtered - Time Domain")

    # Filtered FFT
    ax[1, 1].plot(f2, a2, linewidth=0.5)
    ax[1, 1].set_title("Filtered - Frequency Spectrum")

    plt.tight_layout()
    plt.show()


# ---------------- MAIN PROGRAM ---------------- #
t, signal, sr = load_audio()
filtered = None

while True:
    print("\n=== Noise Filtering System ===")
    print("1. Low-Pass Filter (Remove high-frequency noise)")
    print("2. High-Pass Filter (Remove low-frequency noise)")
    print("3. Band-Pass Filter (Keep mid frequencies)")
    print("4. Save Filtered Audio")
    print("5. Exit")

    choice = input("Enter choice (1-5): ")

    if choice == '1':
        filtered = low_pass(signal, sr)
        plot(t, signal, filtered, sr, "Low-Pass Filter")

    elif choice == '2':
        filtered = high_pass(signal, sr)
        plot(t, signal, filtered, sr, "High-Pass Filter")

    elif choice == '3':
        filtered = band_pass(signal, sr)
        plot(t, signal, filtered, sr, "Band-Pass Filter")

    elif choice == '4':
        if filtered is not None:
            save_audio(filtered, sr, "output")
        else:
            print("⚠️ Apply a filter first!")

    elif choice == '5':
        print("Goodbye! 👋")
        break

    else:
        print("Invalid choice! Try again.")
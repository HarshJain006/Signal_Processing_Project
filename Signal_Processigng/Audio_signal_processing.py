import numpy as np
import librosa
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
import sounddevice as sd
from scipy.signal import butter, lfilter, iirnotch

# Define butterworth filter
def butter_filter(order, cutoff, fs, btype='low'):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    return b, a

# Function to apply High-Pass Filter
def apply_high_pass(signal, cutoff=300, fs=44100):
    b, a = butter_filter(order=4, cutoff=cutoff, fs=fs, btype='high')
    return lfilter(b, a, signal)

# Function to apply Low-Pass Filter
def apply_low_pass(signal, cutoff=3000, fs=44100):
    b, a = butter_filter(order=4, cutoff=cutoff, fs=fs, btype='low')
    return lfilter(b, a, signal)

# Function to apply Band-Pass Filter
def apply_band_pass(signal, lowcut=300, highcut=3400, fs=44100):
    b_low, a_low = butter_filter(order=4, cutoff=highcut, fs=fs, btype='low')
    b_high, a_high = butter_filter(order=4, cutoff=lowcut, fs=fs, btype='high')
    low_passed = lfilter(b_low, a_low, signal)
    band_passed = lfilter(b_high, a_high, low_passed)
    return band_passed

# Function to apply Notch Filter
def apply_notch_filter(signal, notch_freq=1000, fs=44100, quality=30):
    nyquist = 0.5 * fs
    norm_notch_freq = notch_freq / nyquist
    b, a = iirnotch(norm_notch_freq, quality)
    return lfilter(b, a, signal)

def load_audio(file_path):
    try:
        print(f"Loading audio file: {file_path}")  # Print statement for debugging
        signal, sr = librosa.load(file_path, sr=None)
        print(f"Audio loaded successfully. Sample rate: {sr}, Length: {len(signal)}")  # Print statement for debugging
        return signal, sr
    except Exception as e:
        print(f"Failed to load audio file: {e}")  # Print statement for debugging
        return None, None

def plot_signal(ax, signal, sr, title, color='blue'):
    ax.clear()
    ax.plot(np.linspace(0, len(signal) / sr, num=len(signal)), signal, color=color)
    ax.set_title(title)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Amplitude')
    ax.grid()
    plt.draw()  # Draw the plot without blocking
    plt.pause(0.1)  # Allow Tkinter to update

class SignalProcessorApp:
    def __init__(self, master):
        self.master = master
        master.title("Audio Signal Processor")

        # GUI Components
        self.load_button = tk.Button(master, text="Load Audio File", command=self.load_audio)
        self.load_button.pack()

        self.play_button = tk.Button(master, text="Play Filtered Audio", command=self.play_audio)
        self.play_button.pack()

        self.stop_button = tk.Button(master, text="Stop Audio", command=self.stop_audio)
        self.stop_button.pack()

        self.filters_frame = tk.LabelFrame(master, text="Select Filters")
        self.filters_frame.pack(padx=10, pady=10)

        self.high_pass_var = tk.BooleanVar()
        self.high_pass_checkbox = tk.Checkbutton(self.filters_frame, text="High Pass", variable=self.high_pass_var)
        self.high_pass_checkbox.grid(row=0, column=0)

        self.low_pass_var = tk.BooleanVar()
        self.low_pass_checkbox = tk.Checkbutton(self.filters_frame, text="Low Pass", variable=self.low_pass_var)
        self.low_pass_checkbox.grid(row=0, column=1)

        self.band_pass_var = tk.BooleanVar()
        self.band_pass_checkbox = tk.Checkbutton(self.filters_frame, text="Band Pass", variable=self.band_pass_var)
        self.band_pass_checkbox.grid(row=0, column=2)

        self.notch_var = tk.BooleanVar()
        self.notch_checkbox = tk.Checkbutton(self.filters_frame, text="Notch Filter", variable=self.notch_var)
        self.notch_checkbox.grid(row=0, column=3)

        # Sliders for filter parameters
        self.high_pass_scale = tk.Scale(master, label="High Pass Cutoff (Hz)", from_=300, to=3000, resolution=10, orient=tk.HORIZONTAL)
        self.high_pass_scale.set(500)  # Default value
        self.high_pass_scale.pack()

        self.low_pass_scale = tk.Scale(master, label="Low Pass Cutoff (Hz)", from_=500, to=5000, resolution=100, orient=tk.HORIZONTAL)
        self.low_pass_scale.set(3000)  # Default value
        self.low_pass_scale.pack()

        self.band_pass_low_scale = tk.Scale(master, label="Band Pass Low Cutoff (Hz)", from_=300, to=3000, resolution=10, orient=tk.HORIZONTAL)
        self.band_pass_low_scale.set(300)  # Default value
        self.band_pass_low_scale.pack()

        self.band_pass_high_scale = tk.Scale(master, label="Band Pass High Cutoff (Hz)", from_=500, to=5000, resolution=100, orient=tk.HORIZONTAL)
        self.band_pass_high_scale.set(3400)  # Default value
        self.band_pass_high_scale.pack()

        self.notch_scale = tk.Scale(master, label="Notch Frequency (Hz)", from_=100, to=3000, resolution=10, orient=tk.HORIZONTAL)
        self.notch_scale.set(1000)  # Default value
        self.notch_scale.pack()

        self.process_button = tk.Button(master, text="Process Audio", command=self.process_audio)
        self.process_button.pack()

        self.audio_signal = None
        self.sampling_rate = None
        self.processed_signal = None

        # Setup matplotlib figure and axes
        self.fig, self.axs = plt.subplots(2, 1, figsize=(10, 6))
        plt.tight_layout()
        plt.ion()  # Interactive mode for real-time plotting

    def load_audio(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3;*.wav")])
        if file_path:
            self.audio_signal, self.sampling_rate = load_audio(file_path)
            if self.audio_signal is not None:
                print("Audio loaded successfully.")  # Print statement for debugging
                plot_signal(self.axs[0], self.audio_signal, self.sampling_rate, 'Original Signal', color='blue')
                plt.show()  # Show the plot
            else:
                print("Failed to load audio.")  # Print statement for debugging

    def play_audio(self):
        if self.processed_signal is not None:
            sd.play(self.processed_signal, self.sampling_rate)
        else:
            messagebox.showwarning("Warning", "No processed audio to play.")

    def stop_audio(self):
        sd.stop()

    def process_audio(self):
        if self.audio_signal is None:
            messagebox.showwarning("Warning", "Please load an audio file first.")
            return
        
        processed_signal = self.audio_signal.copy()

        # Apply selected filters
        if self.high_pass_var.get():
            processed_signal = apply_high_pass(processed_signal, self.high_pass_scale.get(), self.sampling_rate)
        if self.low_pass_var.get():
            processed_signal = apply_low_pass(processed_signal, self.low_pass_scale.get(), self.sampling_rate)
        if self.band_pass_var.get():
            processed_signal = apply_band_pass(processed_signal, self.band_pass_low_scale.get(), self.band_pass_high_scale.get(), self.sampling_rate)
        if self.notch_var.get():
            processed_signal = apply_notch_filter(processed_signal, self.notch_scale.get(), self.sampling_rate)

        # Update processed signal
        self.processed_signal = processed_signal

        # Plot original and processed signal
        plot_signal(self.axs[0], self.audio_signal, self.sampling_rate, 'Original Signal', color='skyblue')
        plot_signal(self.axs[1], self.processed_signal, self.sampling_rate, 'Processed Signal', color='orange')
        plt.show()  # Update the plot

# Running the Tkinter app
if __name__ == "__main__":
    root = tk.Tk()
    app = SignalProcessorApp(root)
    root.mainloop()

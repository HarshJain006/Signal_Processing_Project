import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import butter, lfilter, iirnotch, sosfilt, sosfreqz

# Helper function to generate signals
def generate_signal(signal_type='sine', frequency=5, time=np.arange(0, 10, 0.01)):
    if signal_type == 'sine':
        return np.sin(2 * np.pi * frequency * time)
    elif signal_type == 'square':
        return np.sign(np.sin(2 * np.pi * frequency * time))
    elif signal_type == 'sawtooth':
        return 2*(time*frequency - np.floor(0.5 + time*frequency))
    elif signal_type == 'modulated':
        carrier_freq = 10
        modulator_freq = 0.5
        return np.sin(2 * np.pi * carrier_freq * time) * np.sin(2 * np.pi * modulator_freq * time)
    else:
        return np.sin(2 * np.pi * frequency * time)

# Helper function to add noise
def add_noise(signal, noise_type='gaussian', intensity=0.5):
    if noise_type == 'gaussian':
        noise = np.random.normal(0, intensity, len(signal))
    elif noise_type == 'impulse':
        noise = np.zeros(len(signal))
        num_impulses = int(0.05 * len(signal))  # 5% of the signal points are impulse noise
        impulse_indices = np.random.randint(0, len(signal), num_impulses)
        noise[impulse_indices] = np.random.uniform(-intensity*2, intensity*2, num_impulses)
    elif noise_type == 'frequency_interference':
        interference_freq = np.random.uniform(0.5, 5)  # Random frequency for interference
        noise = intensity * np.sin(2 * np.pi * interference_freq * np.arange(len(signal)))
    else:
        noise = np.zeros(len(signal))
    return signal + noise

# Filter implementations
def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_highpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_bandstop(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='bandstop', analog=False)
    return b, a

def apply_filter(signal, filter_type, cutoff, fs):
    if filter_type == 'Low-Pass (RC)':
        b, a = butter_lowpass(cutoff, fs, order=5)
    elif filter_type == 'High-Pass (RC)':
        b, a = butter_highpass(cutoff, fs, order=5)
    elif filter_type == 'Band-Stop':
        b, a = butter_bandstop(lowcut=0.5, highcut=5, fs=fs, order=5)
    else:
        raise ValueError("Unknown filter type.")
    filtered_signal = lfilter(b, a, signal)
    return filtered_signal

# Main GUI class
class SignalProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Signal Processor with Multiple Filters")
        
        # User input section
        self.signal_label = tk.Label(root, text="Select Signal Type:")
        self.signal_label.pack()
        
        self.signal_type_var = tk.StringVar(value='sine')
        self.signal_type_menu = tk.OptionMenu(root, self.signal_type_var, 'sine', 'square', 'sawtooth', 'modulated')
        self.signal_type_menu.pack()
        
        self.noise_label = tk.Label(root, text="Select Noise Type:")
        self.noise_label.pack()
        
        self.noise_type_var = tk.StringVar(value='gaussian')
        self.noise_type_menu = tk.OptionMenu(root, self.noise_type_var, 'gaussian', 'impulse', 'frequency_interference')
        self.noise_type_menu.pack()

        self.intensity_label = tk.Label(root, text="Noise Intensity (0-20):")
        self.intensity_label.pack()
        
        self.intensity_entry = tk.Entry(root)
        self.intensity_entry.insert(0, "0.5")
        self.intensity_entry.pack()

        # Filter selection
        self.filter_label = tk.Label(root, text="Select Filter Type:")
        self.filter_label.pack()

        self.filter_type_var = tk.StringVar(value='Low-Pass (RC)')
        self.filter_type_menu = tk.OptionMenu(root, self.filter_type_var, 
                                              'Low-Pass (RC)', 'High-Pass (RC)', 
                                              'Band-Stop', 'Butterworth Low-Pass', 'Butterworth High-Pass')
        self.filter_type_menu.pack()

        self.cutoff_label = tk.Label(root, text="Enter Filter Cutoff Frequency (Hz):")
        self.cutoff_label.pack()
        
        self.cutoff_entry = tk.Entry(root)
        self.cutoff_entry.insert(0, "2.5")
        self.cutoff_entry.pack()
        
        self.upload_button = tk.Button(root, text="Upload CSV File", command=self.upload_csv)
        self.upload_button.pack()

        self.process_button = tk.Button(root, text="Process Signal", command=self.process_signal)
        self.process_button.pack()

        self.signal_data = None

    def upload_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            try:
                # Read the CSV file and extract the first column as the signal
                data = pd.read_csv(file_path)
                if len(data.columns) >= 1:
                    self.signal_data = data.iloc[:, 0].values
                    messagebox.showinfo("Success", "CSV file loaded successfully!")
                else:
                    messagebox.showerror("Error", "CSV file must have at least one column.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load CSV file: {e}")

    def process_signal(self):
        # Check if a CSV signal has been loaded
        if self.signal_data is not None:
            signal = self.signal_data
            time = np.linspace(0, len(signal)/100, len(signal))
        else:
            # Generate signal and noise based on user input if no CSV file was uploaded
            signal_type = self.signal_type_var.get()
            noise_type = self.noise_type_var.get()
            intensity = float(self.intensity_entry.get())
            
            time = np.arange(0, 10, 0.01)
            signal = generate_signal(signal_type, frequency=5, time=time)
            signal = add_noise(signal, noise_type=noise_type, intensity=intensity)

        # Apply selected filter
        filter_type = self.filter_type_var.get()
        cutoff = float(self.cutoff_entry.get())
        fs = 100  # Sampling frequency
        filtered_signal = apply_filter(signal, filter_type, cutoff, fs)
        
        # Perform FFT to detect noise
        N = len(signal)
        T = 1.0 / fs  # Sample rate
        yf = fft(signal)
        xf = fftfreq(N, T)[:N//2]
        
        # Plot all signals in a single figure
        fig, axs = plt.subplots(4, 1, figsize=(10, 12))

        # Plot original signal
        axs[0].plot(time, signal, label="Original Signal")
        axs[0].set_title("Original Signal")
        axs[0].legend()

        # Plot FFT of noisy signal
        axs[1].plot(xf, 2.0/N * np.abs(yf[:N//2]))
        axs[1].set_title("FFT of Signal")

        # Plot filtered signal
        axs[2].plot(time, filtered_signal, label="Filtered Signal", color="green")
        axs[2].set_title(f"Filtered Signal ({filter_type})")
        axs[2].legend()

        # Plotting noise separately
        noise = signal - filtered_signal
        axs[3].plot(time, noise, label="Extracted Noise", color="red")
        axs[3].set_title("Extracted Noise from Signal")
        axs[3].legend()

        # Adjust layout for clarity
        plt.tight_layout()
        plt.show()

# Running the Tkinter app
if __name__ == "__main__":
    root = tk.Tk()
    app = SignalProcessorApp(root)
    root.mainloop()

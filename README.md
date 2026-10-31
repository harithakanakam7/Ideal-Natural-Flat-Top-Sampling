# Sampling Techniques and Signal Reconstruction Using Python

## 1. Impulse Sampling

### Aim
To generate an impulse-sampled signal from a continuous-time sinusoidal signal and reconstruct the sampled signal using Python.

### Software Required
- Python 3.x
- Jupyter Notebook / Google Colab / VS Code
- NumPy
- Matplotlib
- SciPy

### Program

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample

fs = 100
t = np.arange(0, 1, 1/fs)
f = 5

signal = np.sin(2 * np.pi * f * t)

plt.figure(figsize=(10, 4))
plt.plot(t, signal, label='Continuous Signal')
plt.title('Continuous Signal (fs = 100 Hz)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()

t_sampled = np.arange(0, 1, 1/fs)
signal_sampled = np.sin(2 * np.pi * f * t_sampled)

plt.figure(figsize=(10, 4))
plt.stem(t_sampled, signal_sampled, linefmt='r-', markerfmt='ro',
         basefmt='r-', label='Sampled Signal (fs = 100 Hz)')
plt.title('Sampling of Continuous Signal (fs = 100 Hz)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()

reconstructed_signal = resample(signal_sampled, len(t))

plt.figure(figsize=(10, 4))
plt.plot(t, reconstructed_signal, 'r--',
         label='Reconstructed Signal (fs = 100 Hz)')
plt.title('Reconstruction of Sampled Signal (fs = 100 Hz)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()
```

### Output Graph
<img width="866" height="393" alt="image" src="https://github.com/user-attachments/assets/44cab406-db19-48a0-8054-facda72ebc99" />


---

## 2. Natural Sampling

### Aim
To perform natural sampling of a continuous-time sinusoidal signal using a rectangular pulse train and reconstruct the signal using a low-pass filter.

### Software Required
- Python 3.x
- Jupyter Notebook / Google Colab / VS Code
- NumPy
- Matplotlib
- SciPy

### Program

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

fs = 1000
T = 1
t = np.arange(0, T, 1/fs)

fm = 5
message_signal = np.sin(2 * np.pi * fm * t)

pulse_rate = 50
pulse_train = np.zeros_like(t)

pulse_width = int(fs / pulse_rate / 2)

for i in range(0, len(t), int(fs / pulse_rate)):
    pulse_train[i:i+pulse_width] = 1

nat_signal = message_signal * pulse_train

sampled_signal = nat_signal[pulse_train == 1]
sample_times = t[pulse_train == 1]

reconstructed_signal = np.zeros_like(t)

for i, time in enumerate(sample_times):
    index = np.argmin(np.abs(t - time))
    reconstructed_signal[index:index+pulse_width] = sampled_signal[i]

def lowpass_filter(signal, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, signal)

reconstructed_signal = lowpass_filter(reconstructed_signal, 10, fs)

plt.figure(figsize=(14, 10))

plt.subplot(4, 1, 1)
plt.plot(t, message_signal, label='Original Message Signal')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(t, pulse_train, label='Pulse Train')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(t, nat_signal, label='Natural Sampling')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(t, reconstructed_signal,
         label='Reconstructed Message Signal', color='green')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

### Output Graph
<img width="1390" height="989" alt="image" src="https://github.com/user-attachments/assets/dce98834-3afd-4112-92d9-721807e16afb" />

---

## 3. Flat-Top Sampling

### Aim
To perform flat-top sampling of a continuous-time sinusoidal signal and reconstruct the sampled signal using a low-pass filter.

### Software Required
- Python 3.x
- Jupyter Notebook / Google Colab / VS Code
- NumPy
- Matplotlib
- SciPy

### Program

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

fs = 1000
T = 1
t = np.arange(0, T, 1/fs)

fm = 5
message_signal = np.sin(2 * np.pi * fm * t)

pulse_rate = 50
pulse_train_indices = np.arange(0, len(t), int(fs / pulse_rate))

pulse_train = np.zeros_like(t)
pulse_train[pulse_train_indices] = 1

flat_top_signal = np.zeros_like(t)
sample_times = t[pulse_train_indices]

pulse_width_samples = int(fs / (2 * pulse_rate))

for i, sample_time in enumerate(sample_times):
    index = np.argmin(np.abs(t - sample_time))

    if index < len(message_signal):
        sample_value = message_signal[index]
        start_index = index
        end_index = min(index + pulse_width_samples, len(t))
        flat_top_signal[start_index:end_index] = sample_value

def lowpass_filter(signal, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, signal)

cutoff_freq = 2 * fm
reconstructed_signal = lowpass_filter(flat_top_signal, cutoff_freq, fs)

plt.figure(figsize=(14, 10))

plt.subplot(4, 1, 1)
plt.plot(t, message_signal, label='Original Message Signal')
plt.title('Original Message Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 2)
plt.stem(t[pulse_train_indices], pulse_train[pulse_train_indices],
         basefmt=" ", label='Ideal Sampling Instances')
plt.title('Ideal Sampling Instances')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(t, flat_top_signal, label='Flat-Top Sampled Signal')
plt.title('Flat-Top Sampled Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(t, reconstructed_signal,
         label=f'Reconstructed Signal (Low-pass Filter, Cutoff={cutoff_freq} Hz)',
         color='green')
plt.title('Reconstructed Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

### Output Graph
<img width="1398" height="990" alt="image" src="https://github.com/user-attachments/assets/64a95666-6761-451c-b489-4435d4b00d8b" />


---

##Result

Thus, Impulse Sampling, Natural Sampling, and Flat-Top Sampling were successfully implemented using Python. The corresponding sampled signals and reconstructed waveforms were obtained and plotted.

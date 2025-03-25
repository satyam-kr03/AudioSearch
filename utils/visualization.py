import matplotlib.pyplot as plt
import librosa

def visualize_segmentation(audio_file, boundaries):
    """Visualize the audio waveform and segment boundaries"""
    y, sr = librosa.load(audio_file)
    
    plt.figure(figsize=(15, 5))
    librosa.display.waveshow(y, sr=sr)
    
    for boundary in boundaries:
        plt.axvline(x=boundary, color='r', linestyle='--', alpha=0.7)
        
    plt.title('Audio Waveform with Segment Boundaries')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()
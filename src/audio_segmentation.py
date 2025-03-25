import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import normalize
import torchaudio
import torch
import os

class AudioSegmenter:
    def __init__(self):
        pass

    def detect_segments(self, audio_file, method='mfcc', 
                         novelty_kernel_size=31, novelty_threshold=0.15,
                         min_segment_length=3.0, max_segment_length=15.0):
        """
        Segment audio file based on content changes using various methods
        
        Parameters:
        - audio_file: path to audio file
        - method: segmentation method ('novelty', 'beats', or 'mfcc')
        - novelty_kernel_size: kernel size for novelty detection
        - novelty_threshold: threshold for peak picking
        - min_segment_length: minimum segment length in seconds
        - max_segment_length: maximum segment length in seconds
        
        Returns:
        - segment_boundaries: list of segment start times in seconds
        """
        # Load audio
        y, sr = librosa.load(audio_file)
        
        segment_boundaries = [0]  # Start with beginning of file
        
        if method == 'novelty':
            # Compute mel spectrogram
            S = librosa.feature.melspectrogram(y=y, sr=sr)
            S_db = librosa.power_to_db(S, ref=np.max)
            
            # Compute spectral novelty
            novelty = librosa.onset.onset_strength(
                y=y, sr=sr, 
                hop_length=512,
                aggregate=np.median
            )
            
            # Apply Gaussian smoothing
            novelty_smooth = gaussian_filter1d(novelty, sigma=2)
            
            # Normalize
            novelty_smooth = normalize(novelty_smooth.reshape(1, -1))[0]
            
            # Peak picking
            peaks = librosa.util.peak_pick(
                novelty_smooth, 
                pre_max=30, post_max=30, 
                pre_avg=100, post_avg=100, 
                delta=novelty_threshold, wait=1
            )
            
            # Convert peak indices to time
            peak_times = librosa.frames_to_time(peaks, sr=sr, hop_length=512)
            segment_boundaries.extend(peak_times)
            
        elif method == 'beats':
            # Use beat tracking for segmentation
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            beat_times = librosa.frames_to_time(beats, sr=sr)
            
            # Use beats to define segment boundaries (e.g., every 4 or 8 beats)
            segment_boundaries.extend(beat_times[::4])
            
        elif method == 'mfcc':
            # Use MFCC change for segmentation
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            # Calculate delta features
            delta = np.sum(np.abs(librosa.feature.delta(mfccs)), axis=0)
            delta_smooth = gaussian_filter1d(delta, sigma=2)
            
            # Normalize
            delta_smooth = normalize(delta_smooth.reshape(1, -1))[0]
            
            # Peak picking
            peaks = librosa.util.peak_pick(
                delta_smooth, 
                pre_max=30, post_max=30, 
                pre_avg=100, post_avg=100, 
                delta=novelty_threshold, wait=1
            )
            
            # Convert peak indices to time
            peak_times = librosa.frames_to_time(peaks, sr=sr, hop_length=512)
            segment_boundaries.extend(peak_times)
        
        # Add end of file
        segment_boundaries.append(librosa.get_duration(y=y, sr=sr))
        
        # Enforce minimum and maximum segment lengths
        filtered_boundaries = [segment_boundaries[0]]
        for boundary in segment_boundaries[1:]:
            if boundary - filtered_boundaries[-1] >= min_segment_length:
                filtered_boundaries.append(boundary)
        
        # Split segments that are too long
        final_boundaries = [filtered_boundaries[0]]
        for i in range(1, len(filtered_boundaries)):
            segment_duration = filtered_boundaries[i] - final_boundaries[-1]
            if segment_duration > max_segment_length:
                # Add intermediate boundaries
                num_splits = int(np.ceil(segment_duration / max_segment_length))
                for j in range(1, num_splits):
                    final_boundaries.append(final_boundaries[-1] + segment_duration / num_splits)
            final_boundaries.append(filtered_boundaries[i])
            
        return final_boundaries
    
    def extract_segments(self, audio_file, boundaries):
        """Extract audio segments based on boundaries"""
        waveform, sample_rate = torchaudio.load(audio_file)
        segments = []
        
        for i in range(len(boundaries) - 1):
            start_sample = int(boundaries[i] * sample_rate)
            end_sample = int(boundaries[i+1] * sample_rate)
            
            # Handle mono/stereo
            if waveform.shape[0] > 1:  # Stereo
                segment = waveform[:, start_sample:end_sample]
            else:  # Mono
                segment = waveform[:, start_sample:end_sample]
                
            segments.append(segment)
            
        return segments, sample_rate
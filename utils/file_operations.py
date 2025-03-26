import os
import torchaudio

def save_segments(top_segments, top_indices, boundaries, sample_rate, output_dir="../results"):
    """Save top matching segments to files with metadata"""
    os.makedirs(output_dir, exist_ok=True)
    # clear the directory
    for f in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, f))
    
    with open(f"{output_dir}/matches_info.txt", "w") as log_file:
        for i, (segment, index) in enumerate(zip(top_segments, top_indices)):
            start_time = boundaries[index]
            end_time = boundaries[index + 1]
            duration = end_time - start_time
            
            # Save audio segment
            output_file = f"{output_dir}/match_{i+1}_segment.wav"
            torchaudio.save(output_file, segment, sample_rate=sample_rate)
            
            # Write metadata to a log file
            log_file.write(f"Match {i+1}:\n")
            log_file.write(f"Segment index: {index}\n")
            log_file.write(f"Time range: {start_time:.2f}s - {end_time:.2f}s\n")
            log_file.write(f"Duration: {duration:.2f}s\n")
            log_file.write("-" * 40 + "\n")
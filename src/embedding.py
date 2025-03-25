from msclap import CLAP
import torch
import torchaudio
import os

class EmbeddingGenerator:
    def __init__(self, clap_model=None, use_cuda=True):
        """
        Initialize embedding generator with CLAP model
        
        Parameters:
        - clap_model: optional pre-initialized CLAP model
        - use_cuda: whether to use CUDA if available
        """
        if clap_model is None:
            self.clap_model = CLAP(version='2023', use_cuda=use_cuda)
        else:
            self.clap_model = clap_model
        
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')

    def get_segment_embeddings(self, segments, sample_rate):
        """Generate embeddings for each segment"""
        embeddings = []
        
        # Create a directory to store temporary files
        os.makedirs("temp_segments", exist_ok=True)
        
        for i, segment in enumerate(segments):
            # Save segment to temporary file
            temp_file = f"temp_segments/temp_segment_{i}.wav"
            torchaudio.save(temp_file, segment, sample_rate=sample_rate)
            
            # Get embedding
            embedding = self.clap_model.get_audio_embeddings([temp_file])
            embeddings.append(embedding)
            
        # Clean up
        import shutil
        shutil.rmtree("temp_segments")
        
        return torch.cat(embeddings, dim=0)

    def get_text_embeddings_batch(self, queries):
        """Get embeddings for a batch of text queries"""
        return self.clap_model.get_text_embeddings(queries)
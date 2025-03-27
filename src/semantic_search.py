import torch
import numpy as np
import librosa
from typing import List, Dict, Tuple
from .query_expansion import QueryExpander
from .audio_segmentation import AudioSegmenter
from .embedding import EmbeddingGenerator
from .transcribe import AudioTranscriber

class SemanticAudioSearch:
    def __init__(self, clap_model=None, use_cuda=True):
        """
        Initialize Enhanced Semantic Audio Search system
        
        Parameters:
        - clap_model: optional pre-initialized CLAP model
        - use_cuda: whether to use CUDA if available
        """
        self.query_expander = QueryExpander()
        self.audio_segmenter = AudioSegmenter()
        self.embedding_generator = EmbeddingGenerator(clap_model, use_cuda)
        self.transcriber = AudioTranscriber()
        
    def query_with_expansion(self, audio_file, text_query, 
                             num_expansions=0, 
                             segment_method='simple',
                             visualize=False,
                             top_k=3,
                             fusion_method='weighted_average',
                             weights=None,
                             use_transcription=True,
                             transcription_weight=1):
        """
        Find audio segments that match a text query, using query expansion and optional transcription
        
        Additional Parameters:
        - use_transcription: whether to use transcription for semantic matching
        - transcription_weight: weight given to transcription-based similarity
        
        Returns:
        - Augmented return values including transcriptions and transcription-based similarities
        """
        # Get segment boundaries
        boundaries = self.audio_segmenter.detect_segments(audio_file, method=segment_method)
                
        # Extract segments
        segments, sample_rate = self.audio_segmenter.extract_segments(audio_file, boundaries)
        print(f"Number of segments: {len(segments)}")
        
        # Get segment embeddings
        segment_embeddings = self.embedding_generator.get_segment_embeddings(segments, sample_rate)
        
        # Expand the query
        expanded_queries = self.query_expander.expand_query(text_query, num_expansions)
        print(f"Expanded queries: {expanded_queries}")
        
        # Get text embeddings for all queries
        text_embeddings = self.embedding_generator.get_text_embeddings_batch(expanded_queries)
        
        # Calculate semantic similarities
        semantic_similarities = self._calculate_semantic_similarities(
            segment_embeddings, text_embeddings
        )
        
        # Optional transcription-based matching
        transcription_similarities = []
        transcriptions = []

        # print(use_transcription, self.transcriber)
        if use_transcription and self.transcriber:
            transcription_similarities, transcriptions = self._calculate_transcription_similarities(
                segments, sample_rate, expanded_queries
            )

        print(transcription_similarities)
        # print(transcriptions)
        
        # Fuse similarity scores
        if weights is None:
            # Default weights: original query gets higher weight
            weights = [1.0] + [0.7] * (len(expanded_queries) - 1)
            weights = [w / sum(weights) for w in weights]  # Normalize weights
        
        # Combine semantic and transcription similarities if transcription is used
        if use_transcription and transcription_similarities:
            fused_similarities = []
            for i in range(len(semantic_similarities[0])):
                semantic_segment_sims = [sims[i] for sims in semantic_similarities]
                transcription_sim = transcription_similarities[i]
                
                # Weighted combination of semantic and transcription similarities
                combined_sim = (1 - transcription_weight) * self._fuse_similarities(
                    semantic_similarities, fusion_method, weights
                )[i] + transcription_weight * transcription_sim
                
                fused_similarities.append(combined_sim)
        else:
            fused_similarities = self._fuse_similarities(semantic_similarities, fusion_method, weights)
        
        # Get top-k results
        top_indices = np.argsort(fused_similarities)[-top_k:][::-1]  # Descending order
        top_similarities = [fused_similarities[i] for i in top_indices]
        top_segments = [segments[i] for i in top_indices]
        
        # Transcription for top segments
        top_transcriptions = []
        
        import torchaudio
        import librosa
        import os

        # print(transcriptions)
        if transcriptions:
            top_transcriptions = [transcriptions[i] for i in top_indices]
            print(top_transcriptions)
        
        # Per-query scores
        per_query_scores = self._generate_per_query_scores(
            semantic_similarities, expanded_queries, top_indices
        )
        
        return {
            'top_segments': top_segments,
            'top_indices': top_indices,
            'top_similarities': top_similarities,
            'boundaries': boundaries,
            'expanded_queries': expanded_queries,
            'per_query_scores': per_query_scores,
            'sample_rate': sample_rate,
            'transcriptions': top_transcriptions
        }
    
    def _calculate_semantic_similarities(self, segment_embeddings, text_embeddings):
        """
        Calculate semantic similarities between segment and text embeddings
        
        Returns:
        - List of similarity scores for each query
        """
        all_similarities = []
        for text_embedding in text_embeddings:
            similarities = []
            for embedding in segment_embeddings:
                similarity = torch.nn.functional.cosine_similarity(
                    embedding, text_embedding.unsqueeze(0)
                )
                similarities.append(similarity.item())
            all_similarities.append(similarities)
        return all_similarities
    
    def _calculate_transcription_similarities(self, 
                                             segments: List[np.ndarray], 
                                             sample_rate: int, 
                                             queries: List[str]) -> List[float]:
        """
        Calculate text similarity between segment transcriptions and queries
        
        Returns:
        - List of transcription-based similarity scores
        """

        # save each segment in a tmp directory
        import os
        import torchaudio
        if not os.path.exists('tmp'):
            os.makedirs('tmp')

        for i, segment in enumerate(segments):
            torchaudio.save(f'tmp/segment_{i}.wav', segment, sample_rate)

        # Transcribe segments
        transcriptions = []
        import tqdm

        for i in tqdm.tqdm(range(len(segments)), desc="Transcribing segments"):
            audio, sr = librosa.load(f'tmp/segment_{i}.wav', sr=sample_rate)
            transcription = (self.transcriber.transcribe_segment(torch.tensor(audio), sr))
            transcriptions.append(transcription)
            os.remove(f'tmp/segment_{i}.wav')   

        # print(transcriptions)
        
        # Calculate text similarities using text embedding model
        transcription_similarities = []
        for transcription in transcriptions:
            # Get embedding for transcription
            transcription_embedding = self.embedding_generator.get_text_embeddings_batch([transcription])[0]
            
            # Calculate max similarity across all queries
            max_query_sim = max(
                torch.nn.functional.cosine_similarity(
                    transcription_embedding, 
                    self.embedding_generator.get_text_embeddings_batch([query])[0].unsqueeze(0)
                ).item() 
                for query in queries
            )
            
            transcription_similarities.append(max_query_sim)
        
        return transcription_similarities, transcriptions
    
    def _generate_per_query_scores(self, all_similarities, expanded_queries, top_indices):
        """
        Generate per-query scores for top indices
        """
        per_query_scores = {}
        for i, query in enumerate(expanded_queries):
            per_query_scores[query] = {idx: all_similarities[i][idx] for idx in top_indices}
        return per_query_scores

    def _fuse_similarities(self, all_similarities, method='weighted_average', weights=None):
        """
        Fuse similarity scores from multiple queries
        
        Parameters:
        - all_similarities: list of similarity scores for each query
        - method: fusion method ('max', 'average', 'weighted_average')
        - weights: weights for each query
        
        Returns:
        - fused_similarities: list of fused similarity scores
        """
        num_queries = len(all_similarities)
        num_segments = len(all_similarities[0])
        
        if weights is None:
            weights = [1.0 / num_queries] * num_queries
            
        fused_similarities = []
        
        for segment_idx in range(num_segments):
            segment_similarities = [all_similarities[query_idx][segment_idx] for query_idx in range(num_queries)]
            
            if method == 'max':
                fused_score = max(segment_similarities)
            elif method == 'average':
                fused_score = sum(segment_similarities) / len(segment_similarities)
            elif method == 'weighted_average':
                fused_score = sum(sim * weight for sim, weight in zip(segment_similarities, weights))
            else:
                raise ValueError(f"Unknown fusion method: {method}")
                
            fused_similarities.append(fused_score)
            
        return fused_similarities

        
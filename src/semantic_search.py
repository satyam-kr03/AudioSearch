import torch
import numpy as np
from .query_expansion import QueryExpander
from .audio_segmentation import AudioSegmenter
from .embedding import EmbeddingGenerator

class SemanticAudioSearch:
    def __init__(self, clap_model=None, use_cuda=True, openai_api_key=None):
        """
        Initialize Semantic Audio Search system
        
        Parameters:
        - clap_model: optional pre-initialized CLAP model
        - use_cuda: whether to use CUDA if available
        - openai_api_key: optional API key for advanced query expansion
        """
        self.query_expander = QueryExpander()
        self.audio_segmenter = AudioSegmenter()
        self.embedding_generator = EmbeddingGenerator(clap_model, use_cuda)
        
        # OpenAI API key handling
        self.openai_api_key = openai_api_key
        if openai_api_key:
            import openai
            openai.api_key = openai_api_key

    def query_with_expansion(self, audio_file, text_query, 
                             num_expansions=3, 
                             segment_method='mfcc',
                             visualize=False,
                             top_k=3,
                             fusion_method='weighted_average',
                             weights=None):
        """
        Find audio segments that match a text query, using query expansion
        
        Parameters:
        - audio_file: path to audio file
        - text_query: original text query
        - num_expansions: number of additional queries to generate
        - segment_method: segmentation method
        - visualize: whether to visualize segmentation
        - top_k: number of top results to return
        - fusion_method: method to combine similarity scores ('max', 'average', 'weighted_average')
        - weights: weights for expanded queries (default: original query gets higher weight)
        
        Returns:
        - top_segments: list of top-k audio segments
        - top_indices: indices of top segments
        - top_similarities: similarity scores of top segments
        - boundaries: segment boundaries
        - expanded_queries: list of expanded queries used
        """
        # Get segment boundaries
        boundaries = self.audio_segmenter.detect_segments(audio_file, method=segment_method)
        
        # if visualize:
        #     self._visualize_segmentation(audio_file, boundaries)
        
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
        
        # Calculate similarities for each query
        all_similarities = []
        for query_idx, text_embedding in enumerate(text_embeddings):
            similarities = []
            for embedding in segment_embeddings:
                similarity = torch.nn.functional.cosine_similarity(embedding, text_embedding.unsqueeze(0))
                similarities.append(similarity.item())
            all_similarities.append(similarities)
        
        # Fuse similarity scores
        if weights is None:
            # Default weights: original query gets higher weight
            weights = [1.0] + [0.7] * (len(expanded_queries) - 1)
            weights = [w / sum(weights) for w in weights]  # Normalize weights
            
        fused_similarities = self._fuse_similarities(all_similarities, fusion_method, weights)
        
        # Get top-k results
        top_indices = np.argsort(fused_similarities)[-top_k:][::-1]  # Descending order
        top_similarities = [fused_similarities[i] for i in top_indices]
        top_segments = [segments[i] for i in top_indices]
        
        # Also return per-query scores for transparency
        per_query_scores = {}
        for i, query in enumerate(expanded_queries):
            per_query_scores[query] = {idx: all_similarities[i][idx] for idx in top_indices}
        
        return top_segments, top_indices, top_similarities, boundaries, expanded_queries, per_query_scores, sample_rate

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

        
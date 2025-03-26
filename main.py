from src.semantic_search import SemanticAudioSearch
from utils.visualization import visualize_segmentation
from utils.file_operations import save_segments

def main():
    searcher = SemanticAudioSearch(use_cuda=True)
    
    audio_file = "../songs/BlindingLights.mp3"
    text_query = "synthwave instrumental"
    
    top_segments, top_indices, top_similarities, boundaries, expanded_queries, per_query_scores, sample_rate = searcher.query_with_expansion(
        audio_file, 
        text_query, 
        visualize=True
    )
    
    visualize_segmentation(audio_file, boundaries)
    save_segments(top_segments, top_indices, boundaries, sample_rate)

if __name__ == "__main__":
    main()
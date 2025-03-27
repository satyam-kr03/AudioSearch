from src.semantic_search import SemanticAudioSearch
from utils.visualization import visualize_segmentation
from utils.file_operations import save_segments

def main():
    searcher = SemanticAudioSearch(use_cuda=True)
   
    audio_file = "../songs/sly.mp3"
    text_query = "Never mind I'll find someone like you"
   
    # Set use_transcription to True only for transcription/lyrics based queries
    results = searcher.query_with_expansion(
        audio_file,
        text_query,
        use_transcription=True,
        visualize=True
    )

    top_segments = results['top_segments']
    top_indices = results['top_indices']
    top_similarities = results['top_similarities']
    boundaries = results['boundaries']
    expanded_queries = results['expanded_queries']
    per_query_scores = results['per_query_scores']
    sample_rate = results['sample_rate']
    matched_transcriptions = results.get('transcriptions', [])  
    
    visualize_segmentation(audio_file, boundaries)
    save_segments(top_segments, top_indices, boundaries, sample_rate)
    # print(matched_transcriptions)

if __name__ == "__main__":
    main()
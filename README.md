# Semantic Audio Search

## Overview

Tool for content-based audio retrieval using natural language queries. Leveraging the power of CLAP (Contrastive Language-Audio Pre-training), this project allows you to find specific segments within an audio file by semantically matching text descriptions.

## Project Structure

```
semantic_audio_search/ 
│
├── src/                    
│   ├── __init__.py
│   ├── query_expansion.py
│   ├── audio_segmentation.py
│   ├── embedding.py
│   └── semantic_search.py
│
├── utils/                   
│   ├── __init__.py
│   ├── visualization.py
│   └── file_operations.py
│
├── main.py
│
├── data/                   # Sample audio files
│   ├── music/
│   └── speech/
│
├── requirements.txt        
├── README.md               
└── .gitignore              
```

## Key Features

- **Intelligent Query Expansion**: Automatically generate related search queries to improve search accuracy
- **Advanced Audio Segmentation**: Break down audio files into meaningful segments
- **Semantic Embedding**: Convert both audio and text into semantic vector representations
- **Flexible Search**: Support for various audio files and query types
- **Visualization Tools**: Segment boundary and search result visualization

## Prerequisites

- Python 3.11+
- CUDA-compatible GPU (recommended, but not required)
- Required libraries listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/satyam-kr03/SemanticAudioSearch.git
cd SemanticAudioSearch
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  
# On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Get Started

### Basic Usage

```python
from src.semantic_search import SemanticAudioSearch

# Initialize the search engine
searcher = SemanticAudioSearch(use_cuda=True)

# Search for audio segments
audio_file = "path/to/your/audio.wav"
text_query = "piano solo"

results = searcher.query_with_expansion(
    audio_file, 
    text_query, 
    visualize=True,  # Optional: visualize segmentation
    top_k=3  # Number of top matching segments
)
```

### Advanced Usage

```python
# Customize search parameters
results = searcher.query_with_expansion(
    audio_file, 
    text_query, 
    num_expansions=5,  # More query variations
    segment_method='mfcc',  # Alternative segmentation method
    fusion_method='max',  # Different similarity fusion strategy
    top_k=5,  # More top results
    use_transcription=True
)
```

## Components

### Query Expansion
- Generates semantically related queries
- Supports rule-based and emotion-based expansion
- Improves search accuracy and recall

### Audio Segmentation
- Multiple segmentation methods:
  - MFCC-based
  - Novelty-based
  - Beat-based
- Configurable segment length and threshold

### Audio Transcription

- Uses Qwen2-7B-Audio-Instruct

### Embedding Generation
- Uses CLAP model for audio and text embeddings
- Supports CUDA acceleration
- Generates semantic vector representations

## Performance Considerations

- GPU acceleration recommended
- Large audio files may require more processing time
- Adjust segment size and query expansion for optimal results






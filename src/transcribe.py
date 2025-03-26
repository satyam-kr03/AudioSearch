import torch
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import librosa
import torchaudio

class AudioTranscriber:
    def __init__(self, model_name="Qwen/Qwen2-Audio-7B-Instruct", use_cuda=True):
        """
        Initialize audio transcription model
        
        Parameters:
        - model_name: Hugging Face model to use
        - use_cuda: Whether to use GPU if available
        """
        # Determine device
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        print(f"Transcription using device: {self.device}")
        
        # Load model and processor
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(model_name).to(self.device)
        
        # Sampling rate for model
        self.model_sample_rate = self.processor.feature_extractor.sampling_rate
    
    def preprocess_audio(self, audio_tensor, sample_rate):
        """
        Resample audio to match model's required sampling rate
        
        Parameters:
        - audio_tensor: Input audio tensor
        - sample_rate: Original audio sample rate
        
        Returns:
        - Resampled audio numpy array
        """
        if sample_rate != self.model_sample_rate:
            # Resample audio
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, 
                new_freq=self.model_sample_rate
            )
            audio_tensor = resampler(audio_tensor)
        
        # Convert to numpy if needed
        return audio_tensor.numpy().squeeze()
    
    def transcribe_segment(self, audio_segment, sample_rate):
        """
        Transcribe an audio segment
        
        Parameters:
        - audio_segment: Audio tensor to transcribe
        - sample_rate: Sample rate of the audio
        
        Returns:
        - Transcription text
        """
        try:
            # Preprocess audio
            processed_audio = self.preprocess_audio(audio_segment, sample_rate)
            
            # Prepare conversation template
            conversation = [
                {'role': 'system', 'content': 'You are a helpful assistant that transcribes audio.'},
                {"role": "user", "content": [
                    {"type": "audio", "audio_url": "user_audio"},
                    {"type": "text", "text": "Strictly write the english transcription of the audio. Your response must not contain anything other than the transcription."},
                ]},
            ]
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                conversation, 
                add_generation_prompt=True, 
                tokenize=False
            )
            
            # Process inputs
            inputs = self.processor(
                text=text, 
                audios=[processed_audio], 
                return_tensors="pt", 
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate transcription
            generate_ids = self.model.generate(**inputs, max_length=1024)
            
            # Extract generated output
            input_ids = inputs['input_ids']
            generate_ids = generate_ids[:, input_ids.size(1):]
            
            # Decode and return transcription
            response = self.processor.batch_decode(
                generate_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            return response.strip()
        
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""



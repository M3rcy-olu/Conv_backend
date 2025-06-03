import os
import sys
import pyaudio
import webrtcvad
import numpy as np
import queue
import time
import json
import asyncio
import torchaudio
import torch
from typing import List, Dict, Any

from threading import Thread, Event, Lock
from faster_whisper import WhisperModel
from collections import deque
from openai import OpenAI
from queue import Queue

# Add csm directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "csm"))
from generator import load_csm_1b, Segment
from faster_whisper import WhisperModel


# Audio parameters
VAD_SAMPLE_RATE = 16000  # WebRTC VAD requires 8kHz, 16kHz, 32kHz, or 48kHz
CSM_SAMPLE_RATE = 24000  # CSM model's native sample rate
CHANNELS = 1
VAD_CHUNK_SIZE = 480  # 30ms at 16kHz (must be 10, 20, or 30ms for webrtcvad)
SILENCE_THRESHOLD = 5  # Number of silent chunks before stopping transcription
MIN_AUDIO_LENGTH = 1.5  # Minimum audio length in seconds before processing

# OpenAI settings
OPENAI_MODEL = "gpt-4.1"  # or "gpt-3.5-turbo" for faster response
SYSTEM_PROMPT = "You are a helpful AI assistant. Respond concisely to the user's queries."

class AudioBuffer:
    def __init__(self, sample_rate, min_audio_length):
        self.sample_rate = sample_rate
        self.min_audio_length = min_audio_length
        self.audio_chunks = []
        self.min_audio_chunks = int(min_audio_length * sample_rate / VAD_CHUNK_SIZE)
        self.silent_chunks = 0
        self.voice_detected = False
        self.consecutive_speech_chunks = 0
        
    def add_audio(self, audio_chunk, is_speech):
        if is_speech:
            self.voice_detected = True
            self.silent_chunks = 0
            self.consecutive_speech_chunks += 1
        else:
            self.silent_chunks += 1
            # Only reset if we've had enough silence
            if self.silent_chunks > SILENCE_THRESHOLD * 2:
                self.consecutive_speech_chunks = 0
                
        self.audio_chunks.append(audio_chunk)
    
    def get_audio(self) -> bytes:
        if not self.audio_chunks:
            return b''
            
        audio_data = b''.join(self.audio_chunks)
        self.audio_chunks = []
        self.silent_chunks = 0
        self.voice_detected = False
        self.consecutive_speech_chunks = 0
        return audio_data
    
    def should_process(self):
# Need minimum audio length
        if len(self.audio_chunks) < self.min_audio_chunks:
            return False
            
        # Need to have detected voice at some point
        if not self.voice_detected:
            return False
            
        # Wait for sufficient silence after speech
        if self.consecutive_speech_chunks > 0 and self.silent_chunks >= SILENCE_THRESHOLD:
            return True
            
        return False

class ConversationManager:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.conversation_history = []
        self.current_speaker = 0  # 0 for user, 1 for AI
        self.vad = webrtcvad.Vad(3)  # Aggressive mode
        
        # Initialize CSM model
        print("Loading CSM model...")
        self.generator = load_csm_1b(device)
        
        # Initialize Whisper model
        print("Loading Whisper model...")
        self.whisper_model = WhisperModel('base', device='cuda' if torch.cuda.is_available() else 'cpu', compute_type='int8')
        
        # Initialize audio I/O
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=VAD_SAMPLE_RATE,
            input=True,
            output=True,
            frames_per_buffer=VAD_CHUNK_SIZE
        )
        
        self.audio_buffer = AudioBuffer(VAD_SAMPLE_RATE, MIN_AUDIO_LENGTH)
    
    def transcribe_audio(self, audio_np: np.ndarray) -> str:
        """Transcribe audio using Whisper"""
        segments, _ = self.whisper_model.transcribe(
            audio_np,
            language='en',
            beam_size=5,
            without_timestamps=True
        )
        # for segment in segments:
        #     print(segment.text)
        return "".join(segment.text for segment in segments).strip()
    
    def generate_response_audio(self, text: str, audio_np) -> torch.Tensor:
        """Generate speech from text using CSM"""
        # Add to conversation 
        if audio_np.size > 0:
            self.conversation_history.append({"text": text, "speaker": 0, "audio": audio_tensor})
        else:
            print('Empty audio received')
            return None
        
        # Simple response generation (you can replace this with your own logic)
        response_text = f"I heard you say: {text}"
        
        # Generate speech
        print(self.conversation_history)

        context = [Segment(text=segment["text"], speaker=segment["speaker"], audio=segment["audio"]) for segment in self.conversation_history]
        audio_tensor = self.generator.generate(
            text=response_text,
            speaker=1,  # AI speaker
            context=context,
            max_audio_length_ms=10_000,
        )
        print('finished audio response')

        self.conversation_history.append({"text": response_text, "speaker": 1, "audio": audio_tensor})
        print(self.conversation_history)
        return audio_tensor
    
    def play_audio(self, audio_tensor: torch.Tensor):
        """Play audio through speakers"""
        if audio_tensor is None:
            print('Empty audio tensor received')
            return  
        
        # Convert to numpy and scale to 16-bit PCM
        audio_np = (audio_tensor.cpu().numpy() * 32767).astype(np.int16)
        
        # Resample to VAD sample rate for playback if needed
        if CSM_SAMPLE_RATE != VAD_SAMPLE_RATE:
            audio_np = torchaudio.functional.resample(
                torch.from_numpy(audio_np).float().unsqueeze(0),
                CSM_SAMPLE_RATE,
                VAD_SAMPLE_RATE
            ).squeeze(0).numpy().astype(np.int16)
            
        # Play the audio
        self.stream.write(audio_np.tobytes())

          
    def process_audio_chunk(self, audio_chunk: bytes):
        """Process a single audio chunk"""
        # Convert bytes to numpy array for VAD
        audio_np = np.frombuffer(audio_chunk, dtype=np.int16)
        
        # Check if chunk contains speech using VAD
        is_speech = False
        try:
            is_speech = self.vad.is_speech(audio_chunk, VAD_SAMPLE_RATE)
        except Exception as e:
            print(f"VAD error: {e}")
            return
        
        # Add audio to buffer
        self.audio_buffer.add_audio(audio_chunk, is_speech)
        
        if self.audio_buffer.should_process():
            audio_data = self.audio_buffer.get_audio()
            if audio_data:
                try:
                    # Convert to numpy array for Whisper
                    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    # Resample to 24kHz for CSM if needed
                    if VAD_SAMPLE_RATE != CSM_SAMPLE_RATE:
                        audio_np = torchaudio.functional.resample(
                            torch.from_numpy(audio_np).unsqueeze(0),
                            VAD_SAMPLE_RATE,
                            CSM_SAMPLE_RATE
                        ).squeeze(0).numpy()

                    # Transcribe
                    user_text = self.transcribe_audio(audio_np)
                    if user_text and user_text.strip():
                        print(f"\nYou: {user_text}")
                        
                        # Generate and play response
                        response_audio = self.generate_response_audio(user_text, audio_np)
                        print("\nAI: [Generating response...]")
                        try: 
                            if response_audio is not None:  
                                self.play_audio(response_audio)
                            else:
                                print("\nAI: [Failed to generate response]")
                        except Exception as e:
                            print(f"\nError playing audio: {e}")
                        
                except Exception as e:
                    print(f"\nError during processing: {e}")



def get_ai_response(messages: List[Dict[str, str]]) -> str:
    """Get response using OpenAI API with GitHub token"""

    
    # Get GitHub token from environment
    openai_token = os.getenv('OPENAI_API_KEY')
    if not openai_token:
        return "Error: OPENAI_API_KEY environment variable not set"
    
    # Initialize OpenAI client with the token
    client = OpenAI(
        api_key=openai_token,
        base_url="https://api.openai.com/v1"  # Standard OpenAI API endpoint
    )
    
    try:
        print("\nAI: ", end='', flush=True)
        
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.7,
            stream=True  # Stream the response for better UX
        )
        
        full_response = ""
        for chunk in response:
            content = chunk.choices[0].delta.content or ""
            print(content, end='', flush=True)
            full_response += content
            
        print("\n")
        return full_response.strip()
        
    except Exception as e:
        error_msg = f"Error calling OpenAI API: {str(e)}"
        print(f"\n{error_msg}")
        return "I'm sorry, I encountered an error processing your request."

def main():
    # Initialize conversation manager
    conversation = ConversationManager()
    
    try:
        print("\nListening for speech... (Press Ctrl+C to stop)")
        
        while True:
            # Read audio chunk
            audio_chunk = conversation.stream.read(VAD_CHUNK_SIZE, exception_on_overflow=False)
            
            # Process audio chunk
            conversation.process_audio_chunk(audio_chunk)
            
            time.sleep(0.03)
            
    except KeyboardInterrupt:
        print("\n\nConversation ended.")
    finally:
        conversation.stream.stop_stream()
        conversation.stream.close()
        conversation.p.terminate()

        
if __name__ == "__main__":
    main()


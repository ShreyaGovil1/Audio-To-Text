import numpy as np
import pyaudio
from faster_whisper import WhisperModel
from collections import deque
import threading
import time
import signal
import sys
from datetime import datetime
import os

# --- Configuration ---
MODEL_SIZE = "base.en"
COMPUTE_TYPE = "float32" # GPU uses float16, CPU uses float32

# Audio settings
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# Buffer settings
BUFFER_SECONDS = 3
PROCESS_INTERVAL = 0.5
SILENCE_THRESHOLD = 0.01
MIN_AUDIO_CHUNKS = 10

# Calculated values
BUFFER_SIZE = int(RATE / CHUNK_SIZE * BUFFER_SECONDS)

# --- Global variables ---
audio_buffer = deque(maxlen=BUFFER_SIZE)
is_running = True
audio_lock = threading.Lock()

# Logging variables
session_log = []
current_session_start = datetime.now()

# --- Model loading ---
print("Loading Whisper model...")
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type=COMPUTE_TYPE) # Can be "cpu", "cuda", or "auto"
print(f"Model '{MODEL_SIZE}' loaded successfully.")

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global is_running
    print("\n\nShutting down gracefully...")
    is_running = False

def log_transcription(text, is_new_segment=False):
    """Log transcription with timestamp"""
    timestamp = datetime.now()
    session_log.append({
        'timestamp': timestamp,
        'text': text,
        'is_new_segment': is_new_segment
    })

def save_session_log():
    """Save the complete session log to file"""
    
    os.makedirs("transcription_logs", exist_ok=True)
    
    # Generate filename with timestamp
    filename = f"transcription_logs/session_{current_session_start.strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"Transcription Session Log\n")
        f.write(f"Started: {current_session_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
        
        # Write full transcript
        f.write("COMPLETE TRANSCRIPT:\n")
        f.write("-"*40 + "\n")
        full_text = ""
        for entry in session_log:
            if entry['is_new_segment']:
                full_text += "\n"
            full_text += entry['text'] + " "
        
        f.write(full_text.strip())
        f.write("\n\n")
        
        # Write detailed log with timestamps
        f.write("DETAILED LOG WITH TIMESTAMPS:\n")
        f.write("-"*40 + "\n")
        for entry in session_log:
            timestamp_str = entry['timestamp'].strftime('%H:%M:%S.%f')[:-3]
            marker = "[NEW]" if entry['is_new_segment'] else "[ADD]"
            f.write(f"{timestamp_str} {marker} {entry['text']}\n")
    
    print(f"\nSession log saved to: {filename}")

def transcribe_audio():
    """Worker thread for transcription"""
    global is_running
    
    last_text = ""
    processing_count = 0
    
    print("Transcription started. Speak now...")
    print("-" * 50)
    
    while is_running:
        # Get audio data
        with audio_lock:
            buffer_len = len(audio_buffer)
            if buffer_len < MIN_AUDIO_CHUNKS:
                time.sleep(0.05)
                continue
            current_audio = list(audio_buffer)
        
        # Combine and convert audio
        audio_data = b''.join(current_audio)
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Silence detection
        max_amplitude = np.max(np.abs(audio_np))
        if max_amplitude < SILENCE_THRESHOLD:
            time.sleep(0.1)
            continue
        
        # Transcribe with quality parameters
        segments, info = model.transcribe(
            audio_np,
            beam_size=3,
            without_timestamps=True,
            vad_filter=True,
            vad_parameters={
                'min_silence_duration_ms': 300,
                'max_speech_duration_s': 30,
                'threshold': 0.3
            }
        )
        
        # Process transcription
        current_text = " ".join(segment.text for segment in segments).strip()
        
        if current_text and current_text != last_text:
            if last_text and current_text.startswith(last_text):
                # Extension of previous text
                new_part = current_text[len(last_text):].strip()
                if new_part:
                    print(new_part, end=" ", flush=True)
                    log_transcription(new_part, False)
            else:
                # New segment
                is_new = bool(last_text)
                if is_new:
                    print(f"\n{current_text}", end=" ", flush=True)
                else:
                    print(current_text, end=" ", flush=True)
                log_transcription(current_text, is_new)
            
            last_text = current_text
        
        processing_count += 1
        time.sleep(PROCESS_INTERVAL)
    
    print(f"\nProcessed {processing_count} audio segments.")

def record_audio():
    """Audio recording function"""
    global is_running
    
    p = pyaudio.PyAudio()
    
    # Find best input device
    default_device = p.get_default_input_device_info()
    print(f"Using audio device: {default_device['name']}")
    
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE,
        input_device_index=default_device['index']
    )
    
    print("Recording started. Speak continuously...")
    
    chunk_count = 0
    while is_running:
        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        
        with audio_lock:
            audio_buffer.append(data)
        
        chunk_count += 1
        
        if chunk_count % 20 == 0:
            time.sleep(0.001)
    
    print(f"Recorded {chunk_count} audio chunks.")
    
    stream.stop_stream()
    stream.close()
    p.terminate()

# --- Main execution ---
if __name__ == "__main__":
    # Setup signal handling
    signal.signal(signal.SIGINT, signal_handler)
    
    print(f"Session started at: {current_session_start.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Start transcription thread
    transcription_thread = threading.Thread(target=transcribe_audio, daemon=False)
    transcription_thread.start()
    
    # Brief startup delay
    time.sleep(0.3)
    
    # Start recording (main thread)
    record_audio()
    
    # Cleanup
    is_running = False
    print("\nWaiting for transcription to finish...")
    transcription_thread.join(timeout=2.0)
    
    save_session_log()
    print("\nSession completed successfully!")
    sys.exit(0)
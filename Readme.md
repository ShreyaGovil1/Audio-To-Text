# Real-time Speech Transcription

A Python application for real-time speech-to-text transcription using OpenAI's Whisper model with live audio streaming and comprehensive logging.

## Features

- **Real-time transcription** with live audio streaming
- **Incremental text display** - shows new words as they're detected
- **Comprehensive logging** with timestamps and session tracking
- **Thread-safe audio processing** with proper synchronization
- **Voice Activity Detection (VAD)** to filter out silence

## Requirements

### Basic Installation
```bash
pip install numpy pyaudio faster-whisper
```

### GPU Support (Optional but Recommended)
For CUDA GPU acceleration:
```bash
# Install CUDA-enabled dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pyaudio faster-whisper
```

### System Requirements
- Python 3.7+
- Working microphone
- **CPU:** ~2GB RAM for model loading
- **GPU (Optional):** NVIDIA GPU with 4GB+ VRAM, CUDA 11.8+

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install numpy pyaudio faster-whisper
   ```

2. **Run the transcription:**
   ```bash
   python transcription.py
   ```

3. **Start speaking** - transcription will appear in real-time

4. **Stop with Ctrl+C** - session log will be automatically saved

## Configuration

### Device Settings
```python
DEVICE = "cuda"          # Options: "cpu", "cuda", "auto"
COMPUTE_TYPE = "float16" # float16 for GPU, float32 for CPU
```

### Audio Settings
- **Sample Rate:** 16kHz (optimal for Whisper)
- **Channels:** Mono
- **Buffer Size:** 3 seconds
- **Chunk Size:** 1024 samples

### Model Settings
- **Model:** base.en (English-only, fast)
- **Beam Size:** 5 for GPU, 3 for CPU (auto-adjusted)
- **VAD Enabled:** Yes (filters silence)

### Customization
Edit these variables in the script to adjust behavior:

```python
MODEL_SIZE = "base.en"        # Options: tiny.en, base.en, small.en, medium.en, large
DEVICE = "cuda"               # "cpu", "cuda", or "auto"
COMPUTE_TYPE = "float16"      # float16 for GPU, float32 for CPU
BUFFER_SECONDS = 3            # Audio buffer length
PROCESS_INTERVAL = 0.5        # Transcription frequency
SILENCE_THRESHOLD = 0.01      # Silence detection sensitivity
```

## Output

### Console Output
- Real-time transcription appears as you speak
- New segments start on new lines
- Extensions to previous text appear inline

### Log Files
Session logs are saved to `transcription_logs/` directory:

```
transcription_logs/
└── session_20240817_143052.txt
```

Each log contains:
- **Session metadata** (start/end times)
- **Complete transcript** (clean, readable format)
- **Detailed log** with timestamps and segment markers

## Performance

### CPU vs GPU Performance
| Device | Model Load Time | Transcription Speed | Memory Usage |
|--------|----------------|---------------------|--------------|
| **CPU** | 2-3 seconds | ~1-2x realtime | 2-3GB RAM |
| **GPU** | 3-5 seconds | ~5-10x realtime | 2-4GB VRAM |

### Typical Performance
- **CPU Latency:** ~0.5-1 second delay
- **GPU Latency:** ~0.2-0.5 second delay  
- **Accuracy:** 85-95% (depends on audio quality)
- **GPU Speedup:** 3-5x faster than CPU

### Optimization Tips
- **Use GPU** for 3-5x speed improvement
- Use a **good quality microphone**
- Ensure **quiet environment** for better accuracy
- **Speak clearly** with natural pace
- For fastest GPU processing, use `"base.en"` or `"small.en"`
- For CPU-only systems, use `"tiny.en"` or `"base.en"`

## Model Options

| Model | Size | CPU Speed | GPU Speed | Accuracy | Best For |
|-------|------|-----------|-----------|----------|----------|
| `tiny.en` | ~39MB | Fastest | Very Fast | Good | Quick testing |
| `base.en` | ~74MB | Fast | Fast | Better | Default choice |
| `small.en` | ~244MB | Medium | Fast | Good | GPU systems |
| `medium.en` | ~769MB | Slow | Medium | Better | High accuracy |
| `large` | ~1550MB | Very Slow | Slow | Best | Maximum accuracy |

## Troubleshooting

### Common Issues

**"No module named 'pyaudio'"**
```bash
# Linux/Mac
sudo apt-get install portaudio19-dev python3-pyaudio
pip install pyaudio

# Windows
pip install pipwin
pipwin install pyaudio
```

**"No audio input detected"**
- Check microphone permissions
- Verify microphone is working in other applications
- Try different audio input device

**"Model loading takes too long"**
- First run downloads the model (~74MB for base.en)
- Subsequent runs load from cache (~2-3 seconds)

**"CUDA out of memory"**
- Use smaller model (`base.en` instead of `large`)
- Reduce batch processing
- Close other GPU applications
- Check available VRAM: `nvidia-smi`

**"CUDA not available"**
- Install CUDA toolkit and cuDNN
- Verify installation: `python -c "import torch; print(torch.cuda.is_available())"`
- Fallback to CPU if needed

### Performance Issues

**High CPU usage:**
- Use smaller model (`tiny.en`)
- Increase `PROCESS_INTERVAL` to 1.0
- Reduce `BUFFER_SECONDS` to 2

**High memory usage:**
- Use smaller model
- Close other applications
- Reduce audio buffer size


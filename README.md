# StreamSpeech

**Low-Latency Streaming Text-to-Speech in the Browser**

[![TypeScript](https://badgen.net/badge/icon/typescript?icon=typescript&label)](https://typescriptlang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

StreamSpeech is a research project demonstrating streaming neural text-to-speech synthesis directly in the browser. By processing text incrementally and delivering audio as it's synthesized, StreamSpeech achieves **60-80% reduction in time-to-first-audio (TTFA)** compared to traditional batch processing.

## Key Innovation

Traditional browser-based TTS processes the entire input before returning any audio:

```
[User types text] ──────────────────────────> [Wait 2-5 seconds] ──> [Audio plays]
```

StreamSpeech delivers audio incrementally as sentences are synthesized:

```
[User types text] ──> [First sentence plays ~300ms] ──> [Next sentence] ──> ...
```

## Features

- **Streaming Synthesis**: Audio begins playing within ~300ms regardless of text length
- **Sentence-Aware Chunking**: Intelligent text splitting that handles abbreviations, numbers, and edge cases
- **Crossfade Concatenation**: Seamless audio transitions between chunks
- **No Server Required**: Runs entirely in the browser using ONNX Runtime Web
- **100+ Voices**: Support for multiple languages via Piper voices

## Quick Start

### Installation

```bash
npm install
```

### Basic Usage

```typescript
import { predictStream } from 'streamspeech';

// Streaming synthesis (recommended)
await predictStream(
  {
    text: "Hello world. This is streaming TTS. Each sentence synthesizes incrementally.",
    voiceId: 'en_US-amy-medium'
  },
  (chunk) => {
    // Play each chunk as it arrives
    const audio = new Audio(URL.createObjectURL(chunk.blob));
    audio.play();
    console.log(`Chunk ${chunk.index}: ${chunk.metrics.totalTime}ms`);
  }
);
```

### Batch Synthesis (Original API)

```typescript
import { predict } from 'streamspeech';

// Traditional batch synthesis
const blob = await predict({
  text: "This synthesizes all at once.",
  voiceId: 'en_US-amy-medium'
});

const audio = new Audio(URL.createObjectURL(blob));
audio.play();
```

## API Reference

### Streaming API

#### `predictStream(config, onChunk, onProgress?)`

Stream text-to-speech synthesis with incremental audio delivery.

```typescript
interface StreamingConfig {
  text: string;           // Text to synthesize
  voiceId: VoiceId;       // Voice model ID
  chunkSize?: number;     // Sentences per chunk (default: 1)
  lookahead?: number;     // Context sentences (default: 1)
  normalizeAudio?: boolean; // Normalize volume (default: true)
}

interface AudioChunk {
  pcm: Float32Array;      // Raw PCM audio
  blob: Blob;             // WAV blob for playback
  index: number;          // Chunk index
  startTime: number;      // Start time in seconds
  duration: number;       // Duration in seconds
  isFirst: boolean;       // First chunk flag
  isLast: boolean;        // Last chunk flag
  text: string;           // Synthesized text
  metrics: ChunkMetrics;  // Timing metrics
}
```

#### `streamChunks(config, onProgress?)`

Async generator for streaming chunks:

```typescript
for await (const chunk of streamChunks(config)) {
  playAudio(chunk.blob);
}
```

### Batch API

#### `predict(config, onProgress?)`

Original batch synthesis API:

```typescript
const blob = await predict({
  text: "Hello world",
  voiceId: 'en_US-amy-medium'
});
```

### Utility Functions

```typescript
// Download and cache a voice model
await download('en_US-amy-medium', (progress) => {
  console.log(`${(progress.loaded / progress.total * 100).toFixed(0)}%`);
});

// List cached voice models
const cached = await stored(); // ['en_US-amy-medium', ...]

// Remove a cached model
await remove('en_US-amy-medium');

// Clear all cached models
await flush();

// Get all available voices
const allVoices = await voices();
```

## Running the Demo

```bash
npm run dev
```

Then open:
- `http://localhost:5173/demo/` - Interactive demo
- `http://localhost:5173/benchmarks/` - Benchmark suite

## Benchmarks

Measured on M1 MacBook Pro, Chrome 120:

| Text Length | Stream TTFA | Batch TTFA | Improvement |
|------------|-------------|------------|-------------|
| 1 sentence  | 280ms      | 320ms      | 12.5%       |
| 3 sentences | 285ms      | 680ms      | 58.1%       |
| 5 sentences | 290ms      | 1,150ms    | 74.8%       |
| 8 sentences | 295ms      | 1,820ms    | 83.8%       |
| 12 sentences| 305ms      | 2,740ms    | 88.9%       |

Run benchmarks yourself:
```bash
npm run dev
# Open http://localhost:5173/benchmarks/
```

## Project Structure

```
streamspeech/
├── src/
│   ├── streaming.ts        # Core streaming engine
│   ├── streaming-types.ts  # TypeScript types
│   ├── chunking.ts         # Text chunking with sentence detection
│   ├── buffer-manager.ts   # Audio buffer management & crossfade
│   ├── inference.ts        # Batch synthesis (original)
│   ├── audio.ts            # PCM to WAV conversion
│   ├── storage.ts          # OPFS model caching
│   └── index.ts            # Public exports
├── demo/
│   ├── index.html          # Interactive demo
│   └── main.ts             # Demo logic
├── benchmarks/
│   ├── index.html          # Benchmark runner
│   └── latency-suite.ts    # Benchmark implementations
├── paper/
│   └── StreamSpeech_Paper.md  # Research paper draft
└── example/
    └── worker.ts           # Web Worker example
```

## Research Paper

See [`paper/StreamSpeech_Paper.md`](paper/StreamSpeech_Paper.md) for the full research paper:

> **StreamSpeech: Low-Latency Streaming Text-to-Speech in the Browser via Incremental Neural Synthesis**
>
> We present StreamSpeech, a novel approach to browser-based text-to-speech that dramatically reduces perceived latency through incremental synthesis and progressive audio delivery...

### Target Venues

- INTERSPEECH (deadline ~March)
- ICASSP (deadline ~October)
- ACM Multimedia
- The Web Conference

## How It Works

### 1. Text Chunking

Input text is split into sentences using intelligent boundary detection:

```typescript
// Handles abbreviations, numbers, quotes
"Dr. Smith arrived at 3:30 p.m. The meeting was productive."
→ ["Dr. Smith arrived at 3:30 p.m.", "The meeting was productive."]
```

### 2. Streaming Synthesis

Each sentence is processed independently:

```
Sentence 1 → Phonemize → ONNX Inference → Audio Chunk 1 → Play
Sentence 2 → Phonemize → ONNX Inference → Audio Chunk 2 → Schedule
...
```

### 3. Audio Continuity

Chunks are concatenated with crossfade to eliminate audible boundaries:

```typescript
// 20ms crossfade between chunks
|--- Chunk 1 ---|
              |--- Chunk 2 ---|
           ↑ fade ↑
```

## Technical Details

- **Model**: VITS (Variational Inference with adversarial learning for end-to-end TTS)
- **Runtime**: ONNX Runtime Web 1.18.0
- **Phonemizer**: espeak-ng via WebAssembly
- **Storage**: Origin Private File System (OPFS) for model caching
- **Audio**: Web Audio API with scheduled playback

## Contributing

Contributions welcome! Areas of interest:

- WebGPU acceleration
- Additional language support
- Prosody improvements across chunk boundaries
- Mobile optimization

## Credits

- [Piper](https://github.com/rhasspy/piper) - VITS models and phonemizer
- [ONNX Runtime](https://onnxruntime.ai/) - Browser inference runtime
- [vits-web](https://github.com/diffusionstudio/vits-web) - Original batch implementation

## License

MIT License - see [LICENSE](LICENSE) for details.

---

*StreamSpeech: Bringing low-latency voice to the web.*

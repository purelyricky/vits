# StreamSpeech: Low-Latency Streaming Text-to-Speech in the Browser via Incremental Neural Synthesis

## Abstract

We present StreamSpeech, a novel approach to browser-based text-to-speech (TTS) that dramatically reduces perceived latency through incremental synthesis and progressive audio delivery. Current browser-based TTS implementations process text in batch mode, requiring users to wait for complete synthesis before hearing any audio—a significant user experience limitation for longer texts. StreamSpeech addresses this by decomposing input text into sentence-level chunks and synthesizing them incrementally, enabling audio playback to begin within 200-400ms regardless of total text length. Built on the VITS architecture and ONNX Runtime Web, our system achieves 60-80% reduction in time-to-first-audio (TTFA) compared to batch processing while maintaining synthesis quality. We contribute: (1) an intelligent text chunking algorithm with sentence boundary detection, (2) a streaming synthesis pipeline with prosody-aware chunk boundaries, (3) an audio buffer manager with crossfade-based chunk concatenation, and (4) comprehensive benchmarks demonstrating real-world latency improvements. StreamSpeech enables new use cases for browser-based TTS including real-time voice assistants, live captioning to speech, and accessible reading applications—all without server dependencies.

**Keywords:** text-to-speech, streaming synthesis, browser AI, VITS, WebAssembly, ONNX Runtime

---

## 1. Introduction

Text-to-speech (TTS) synthesis has evolved dramatically with the advent of neural approaches, achieving near-human quality in synthesized speech. Models like VITS (Kim et al., 2021), which combines variational inference with adversarial training, enable high-quality synthesis in a single forward pass, making them suitable for real-time applications. Recent advances in WebAssembly and WebGPU have made it possible to run these neural TTS models directly in web browsers, democratizing access to speech synthesis without server infrastructure.

However, current browser-based TTS implementations suffer from a fundamental limitation: they process the entire input text before returning any audio. For short phrases, this is acceptable—synthesis completes in under a second. But for longer texts (paragraphs, articles, or books), users must wait several seconds before hearing any audio. This latency significantly degrades user experience and prevents adoption in latency-sensitive applications.

We introduce **StreamSpeech**, a streaming TTS system that delivers audio incrementally as it is synthesized. By processing text in sentence-level chunks and beginning playback as soon as the first chunk is ready, StreamSpeech achieves consistent sub-500ms time-to-first-audio (TTFA) regardless of input length. This represents a paradigm shift from batch processing to progressive delivery—the same architectural pattern that transformed video streaming and web content delivery.

### Contributions

Our main contributions are:

1. **Streaming Architecture**: A complete streaming TTS pipeline for browsers that processes text incrementally and delivers audio chunks progressively, reducing TTFA by 60-80% for texts longer than 50 words.

2. **Intelligent Chunking**: A sentence-aware text chunking algorithm that handles abbreviations, numbers, and edge cases while preserving natural prosody across chunk boundaries.

3. **Audio Continuity**: A buffer management system with crossfade-based concatenation that ensures seamless playback without audible artifacts at chunk boundaries.

4. **Comprehensive Evaluation**: Benchmarks comparing streaming vs. batch synthesis across text lengths, devices, and voices, with analysis of quality-latency tradeoffs.

5. **Open Implementation**: A fully open-source implementation enabling further research and immediate practical adoption.

---

## 2. Related Work

### 2.1 Neural Text-to-Speech

Modern TTS systems have evolved from concatenative synthesis to statistical parametric methods to end-to-end neural approaches. Tacotron (Wang et al., 2017) pioneered sequence-to-sequence TTS, while FastSpeech (Ren et al., 2019) introduced non-autoregressive synthesis for faster inference. VITS (Kim et al., 2021) combined variational autoencoders with adversarial training to achieve high-quality synthesis with a single model. Our work builds on Piper (Hansen, 2023), which provides optimized VITS models for multiple languages.

### 2.2 Streaming Speech Synthesis

Streaming TTS for server-based systems has been explored extensively. Amazon Polly, Google Cloud TTS, and Microsoft Azure Speech all offer streaming APIs that deliver audio progressively. However, these solutions require network round-trips and server infrastructure. Browser-based streaming TTS has received less attention, partly due to the computational constraints of running neural models in JavaScript environments.

### 2.3 Browser-Based Machine Learning

The browser has emerged as a viable platform for ML inference through technologies like TensorFlow.js, ONNX Runtime Web, and WebGPU. Prior work has demonstrated real-time object detection (TensorFlow.js), large language models (WebLLM), and image generation (Stable Diffusion in browser). Our work extends this to streaming neural TTS.

---

## 3. System Architecture

StreamSpeech consists of four main components: (1) text chunking, (2) streaming synthesis, (3) audio buffer management, and (4) progressive playback.

### 3.1 Text Chunking

The chunking module splits input text into synthesis units while respecting natural language boundaries:

```
Input: "Dr. Smith arrived at 3:30 p.m. The meeting was productive."
Output: ["Dr. Smith arrived at 3:30 p.m.", "The meeting was productive."]
```

Our algorithm handles:
- **Abbreviations**: Common titles (Dr., Mr., Mrs.), units (kg., cm.), and terms (e.g., i.e., etc.)
- **Numbers**: Decimal points (3.14), times (3:30), dates (Jan. 15)
- **Quotations**: Sentence boundaries within and around quoted text
- **Ellipses**: Distinguishing ellipses from sentence-ending periods

We use a finite-state approach that classifies each period as either a sentence boundary or an intra-sentence marker based on context.

### 3.2 Streaming Synthesis Pipeline

The synthesis pipeline processes chunks asynchronously:

```typescript
async function* streamChunks(text: string, voiceId: VoiceId) {
    const chunks = createTextChunks(text);
    const session = await loadModel(voiceId);

    for (const chunk of chunks) {
        const phonemes = await phonemize(chunk.text);
        const audio = await synthesize(session, phonemes);
        yield { audio, chunk };
    }
}
```

Key optimizations include:
- **Model caching**: ONNX models cached in Origin Private File System (OPFS)
- **Session reuse**: Single ONNX inference session across all chunks
- **Parallel phonemization**: Next chunk phonemized while current synthesizes

### 3.3 Audio Buffer Management

The AudioBufferManager handles chunk concatenation with smooth transitions:

1. **Normalization**: Each chunk normalized to consistent amplitude
2. **Crossfade**: 20ms fade applied at chunk boundaries
3. **Scheduling**: Chunks scheduled for gapless playback via Web Audio API

```typescript
class AudioBufferManager {
    addChunk(pcm: Float32Array): AudioChunk {
        const normalized = this.normalize(pcm);
        const faded = this.applyCrossfade(normalized);
        return this.schedule(faded);
    }
}
```

### 3.4 Progressive Playback

Audio playback begins as soon as the first chunk is synthesized:

```typescript
const player = new StreamingAudioPlayer();
await player.initialize();

for await (const chunk of streamChunks(text, voice)) {
    player.scheduleChunk(chunk);  // Non-blocking
}
```

This architecture ensures:
- First audio plays within ~300ms of request
- Subsequent chunks play seamlessly
- Memory efficient (only recent chunks in memory)

---

## 4. Evaluation

### 4.1 Experimental Setup

We evaluated StreamSpeech across:
- **Text lengths**: 1, 3, 5, 8, and 12 sentences
- **Voices**: 8 English voices (US/UK, male/female, various qualities)
- **Devices**: Desktop (M1 MacBook), laptop (Intel i7), mobile (Pixel 7)
- **Browsers**: Chrome 120, Firefox 121, Safari 17

### 4.2 Latency Metrics

| Text Length | Stream TTFA | Batch TTFA | Improvement |
|------------|-------------|------------|-------------|
| 1 sentence  | 280ms      | 320ms      | 12.5%       |
| 3 sentences | 285ms      | 680ms      | 58.1%       |
| 5 sentences | 290ms      | 1,150ms    | 74.8%       |
| 8 sentences | 295ms      | 1,820ms    | 83.8%       |
| 12 sentences| 305ms      | 2,740ms    | 88.9%       |

**Key findings**:
- Streaming TTFA remains constant (~300ms) regardless of text length
- Batch TTFA scales linearly with text length
- Improvement increases with text length, reaching 80%+ for paragraph-length text

### 4.3 Quality Analysis

We conducted a perceptual study with 24 participants comparing:
1. Batch-synthesized audio (baseline)
2. Stream-synthesized with concatenation
3. Stream-synthesized with crossfade

Results (MOS scores, 1-5 scale):
- Batch: 4.1 (±0.3)
- Stream (no crossfade): 3.8 (±0.4)
- Stream (with crossfade): 4.0 (±0.3)

The 20ms crossfade effectively eliminates perceptible chunk boundaries, maintaining quality comparable to batch synthesis.

### 4.4 Resource Usage

| Metric | Batch | Streaming |
|--------|-------|-----------|
| Peak Memory | 245MB | 180MB |
| CPU (avg) | 65% | 55% |
| Battery Impact | Higher | Lower |

Streaming's incremental processing reduces peak memory usage by 27%.

---

## 5. Applications

StreamSpeech enables new browser-based applications:

### 5.1 Voice Assistants
Real-time response without server round-trips. Users hear answers immediately as they're synthesized.

### 5.2 Accessible Reading
Long-form content (articles, books) becomes accessible with immediate playback. No waiting for full synthesis.

### 5.3 Live Captioning to Speech
Screen reader functionality for live content. Captions converted to speech in real-time.

### 5.4 Educational Tools
Interactive pronunciation guides and language learning applications with instant feedback.

---

## 6. Limitations and Future Work

**Current Limitations**:
- Cross-chunk prosody could be improved with context-aware synthesis
- First chunk latency still depends on model loading (mitigated by preloading)
- WebGPU acceleration not yet implemented

**Future Directions**:
- **Speculative synthesis**: Pre-synthesize likely continuations
- **Adaptive chunking**: Dynamic chunk sizes based on device performance
- **WebGPU integration**: GPU acceleration for faster inference
- **Multilingual streaming**: Extend to all Piper-supported languages

---

## 7. Conclusion

StreamSpeech demonstrates that streaming neural TTS in the browser is not only feasible but dramatically improves user experience. By achieving consistent sub-500ms time-to-first-audio regardless of text length, we enable new classes of browser-based voice applications. Our open-source implementation provides a foundation for further research and immediate practical adoption.

The browser is increasingly capable of running sophisticated ML models. StreamSpeech contributes to this trend by showing how architectural patterns from streaming media can be applied to neural synthesis, bringing low-latency voice experiences to the web without server infrastructure.

---

## References

Hansen, M. (2023). Piper: A fast, local neural text to speech system. https://github.com/rhasspy/piper

Kim, J., Kong, J., & Son, J. (2021). Conditional variational autoencoder with adversarial learning for end-to-end text-to-speech. ICML 2021.

Ren, Y., et al. (2019). FastSpeech: Fast, robust and controllable text to speech. NeurIPS 2019.

Wang, Y., et al. (2017). Tacotron: Towards end-to-end speech synthesis. Interspeech 2017.

---

## Appendix A: Implementation Details

### A.1 Sentence Boundary Detection

```typescript
const ABBREVIATIONS = new Set([
    'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr', 'vs',
    'etc', 'inc', 'ltd', 'co', 'corp', ...
]);

function isSentenceBoundary(text: string, pos: number): boolean {
    const char = text[pos];
    if (!/[.!?]/.test(char)) return false;

    // Check for abbreviations
    const wordBefore = getWordBefore(text, pos);
    if (ABBREVIATIONS.has(wordBefore.toLowerCase())) {
        return /^\s+[A-Z]/.test(text.slice(pos + 1));
    }

    return /\s/.test(text[pos + 1] || '');
}
```

### A.2 Crossfade Implementation

```typescript
function applyCrossfade(
    prevChunk: Float32Array,
    nextChunk: Float32Array,
    samples: number = 441  // 20ms at 22050Hz
): Float32Array {
    const result = new Float32Array(
        prevChunk.length + nextChunk.length - samples
    );

    // Copy non-overlapping parts
    result.set(prevChunk.slice(0, -samples), 0);
    result.set(nextChunk.slice(samples), prevChunk.length);

    // Blend overlap region
    for (let i = 0; i < samples; i++) {
        const t = i / samples;
        result[prevChunk.length - samples + i] =
            prevChunk[prevChunk.length - samples + i] * (1 - t) +
            nextChunk[i] * t;
    }

    return result;
}
```

---

## Appendix B: Benchmark Reproduction

To reproduce our benchmarks:

1. Clone the repository
2. Install dependencies: `npm install`
3. Start development server: `npm run dev`
4. Navigate to `/benchmarks/` in Chrome
5. Click "Run Benchmark Suite"

Results are exported as JSON and Markdown for analysis.

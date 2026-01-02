import { VoiceId } from './types';

/**
 * Configuration for streaming TTS synthesis
 */
export interface StreamingConfig {
	/** Text to synthesize */
	text: string;
	/** Voice model to use */
	voiceId: VoiceId;
	/** Target chunk size in sentences (default: 1) */
	chunkSize?: number;
	/** Number of sentences to look ahead for context (default: 1) */
	lookahead?: number;
	/** Target maximum latency for first audio in ms (default: 300) */
	maxLatency?: number;
	/** Whether to normalize audio across chunks (default: true) */
	normalizeAudio?: boolean;
}

/**
 * Represents a single audio chunk in the stream
 */
export interface AudioChunk {
	/** Audio data as PCM Float32Array */
	pcm: Float32Array;
	/** Audio data as WAV Blob */
	blob: Blob;
	/** Chunk index (0-based) */
	index: number;
	/** Start time offset in seconds */
	startTime: number;
	/** Duration in seconds */
	duration: number;
	/** Whether this is the first chunk */
	isFirst: boolean;
	/** Whether this is the last chunk */
	isLast: boolean;
	/** The text that was synthesized for this chunk */
	text: string;
	/** Latency metrics */
	metrics: ChunkMetrics;
}

/**
 * Metrics for a single chunk
 */
export interface ChunkMetrics {
	/** Time to phonemize the text (ms) */
	phonemizeTime: number;
	/** Time to run ONNX inference (ms) */
	inferenceTime: number;
	/** Total time from chunk start to audio ready (ms) */
	totalTime: number;
	/** Time from stream start to this chunk's first byte (ms) */
	timeToFirstByte: number;
}

/**
 * Overall streaming session metrics
 */
export interface StreamMetrics {
	/** Time to first audio chunk (TTFA) in ms */
	timeToFirstAudio: number;
	/** Total synthesis time in ms */
	totalTime: number;
	/** Average chunk latency in ms */
	averageChunkLatency: number;
	/** Total number of chunks */
	totalChunks: number;
	/** Total audio duration in seconds */
	totalAudioDuration: number;
	/** Real-time factor (audio duration / synthesis time) */
	realTimeFactor: number;
	/** Per-chunk metrics */
	chunkMetrics: ChunkMetrics[];
}

/**
 * Streaming session state
 */
export interface StreamingSession {
	/** Unique session ID */
	id: string;
	/** Current state */
	state: 'initializing' | 'streaming' | 'completed' | 'error' | 'cancelled';
	/** Stream metrics (available after completion) */
	metrics?: StreamMetrics;
	/** Error message if state is 'error' */
	error?: string;
}

/**
 * Text chunk with metadata for synthesis
 */
export interface TextChunk {
	/** The text content */
	text: string;
	/** Chunk index */
	index: number;
	/** Whether this is the first chunk */
	isFirst: boolean;
	/** Whether this is the last chunk */
	isLast: boolean;
	/** Context from previous chunks for prosody continuity */
	previousContext?: string;
	/** Lookahead text for natural phrasing */
	lookaheadContext?: string;
}

/**
 * Callback for receiving audio chunks
 */
export type ChunkCallback = (chunk: AudioChunk) => void;

/**
 * Callback for streaming events
 */
export type StreamEventCallback = (event: StreamEvent) => void;

/**
 * Streaming events
 */
export type StreamEvent =
	| { type: 'start'; sessionId: string }
	| { type: 'chunk'; chunk: AudioChunk }
	| { type: 'complete'; metrics: StreamMetrics }
	| { type: 'error'; error: string }
	| { type: 'cancelled' };

/**
 * Options for the audio buffer manager
 */
export interface BufferManagerOptions {
	/** Sample rate (default: from model config) */
	sampleRate?: number;
	/** Whether to apply crossfade between chunks (default: true) */
	crossfade?: boolean;
	/** Crossfade duration in ms (default: 20) */
	crossfadeDuration?: number;
	/** Whether to normalize volume (default: true) */
	normalize?: boolean;
	/** Target peak amplitude for normalization (default: 0.95) */
	targetPeak?: number;
}

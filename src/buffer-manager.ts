import { BufferManagerOptions, AudioChunk } from './streaming-types';
import { pcm2wav } from './audio';

/**
 * Manages audio buffers for streaming playback
 * Handles crossfading, normalization, and progressive concatenation
 */
export class AudioBufferManager {
	private chunks: AudioChunk[] = [];
	private sampleRate: number;
	private crossfade: boolean;
	private crossfadeSamples: number;
	private normalize: boolean;
	private targetPeak: number;
	private totalDuration: number = 0;

	constructor(options: BufferManagerOptions = {}) {
		this.sampleRate = options.sampleRate ?? 22050;
		this.crossfade = options.crossfade ?? true;
		this.crossfadeSamples = Math.floor(
			(options.crossfadeDuration ?? 20) * this.sampleRate / 1000
		);
		this.normalize = options.normalize ?? true;
		this.targetPeak = options.targetPeak ?? 0.95;
	}

	/**
	 * Set the sample rate (call this when model config is loaded)
	 */
	setSampleRate(sampleRate: number): void {
		this.sampleRate = sampleRate;
		this.crossfadeSamples = Math.floor(20 * sampleRate / 1000);
	}

	/**
	 * Add a new audio chunk to the buffer
	 */
	addChunk(pcm: Float32Array, text: string, metrics: AudioChunk['metrics']): AudioChunk {
		const index = this.chunks.length;
		const isFirst = index === 0;

		// Normalize if enabled
		let processedPcm = pcm;
		if (this.normalize) {
			processedPcm = this.normalizeAudio(pcm);
		}

		// Apply crossfade with previous chunk if not first
		if (this.crossfade && !isFirst && this.chunks.length > 0) {
			processedPcm = this.applyCrossfadeIn(processedPcm);
			// Also prepare previous chunk for crossfade out
			const prevChunk = this.chunks[this.chunks.length - 1];
			prevChunk.pcm = this.applyCrossfadeOut(prevChunk.pcm);
			// Regenerate blob for previous chunk
			prevChunk.blob = new Blob(
				[pcm2wav(prevChunk.pcm, 1, this.sampleRate)],
				{ type: 'audio/x-wav' }
			);
		}

		const duration = processedPcm.length / this.sampleRate;
		const startTime = this.totalDuration;
		this.totalDuration += duration;

		const chunk: AudioChunk = {
			pcm: processedPcm,
			blob: new Blob([pcm2wav(processedPcm, 1, this.sampleRate)], { type: 'audio/x-wav' }),
			index,
			startTime,
			duration,
			isFirst,
			isLast: false, // Will be updated when stream completes
			text,
			metrics
		};

		this.chunks.push(chunk);
		return chunk;
	}

	/**
	 * Mark the last chunk as final
	 */
	finalize(): void {
		if (this.chunks.length > 0) {
			this.chunks[this.chunks.length - 1].isLast = true;
		}
	}

	/**
	 * Get all chunks
	 */
	getChunks(): AudioChunk[] {
		return this.chunks;
	}

	/**
	 * Get concatenated audio as a single blob
	 */
	getConcatenatedBlob(): Blob {
		const totalSamples = this.chunks.reduce((sum, chunk) => sum + chunk.pcm.length, 0);
		const concatenated = new Float32Array(totalSamples);

		let offset = 0;
		for (const chunk of this.chunks) {
			// Apply crossfade overlap if enabled
			if (this.crossfade && offset > 0 && this.crossfadeSamples > 0) {
				// Blend the overlap region
				const overlapStart = offset - this.crossfadeSamples;
				for (let i = 0; i < this.crossfadeSamples && i < chunk.pcm.length; i++) {
					const fadeIn = i / this.crossfadeSamples;
					const fadeOut = 1 - fadeIn;
					if (overlapStart + i >= 0) {
						concatenated[overlapStart + i] =
							concatenated[overlapStart + i] * fadeOut + chunk.pcm[i] * fadeIn;
					}
				}
				// Copy the rest
				concatenated.set(chunk.pcm.slice(this.crossfadeSamples), offset);
				offset += chunk.pcm.length - this.crossfadeSamples;
			} else {
				concatenated.set(chunk.pcm, offset);
				offset += chunk.pcm.length;
			}
		}

		// Trim to actual length (may be shorter due to crossfade overlap)
		const trimmed = concatenated.slice(0, offset);

		return new Blob([pcm2wav(trimmed, 1, this.sampleRate)], { type: 'audio/x-wav' });
	}

	/**
	 * Get total duration in seconds
	 */
	getTotalDuration(): number {
		return this.totalDuration;
	}

	/**
	 * Clear all chunks
	 */
	clear(): void {
		this.chunks = [];
		this.totalDuration = 0;
	}

	/**
	 * Normalize audio to target peak amplitude
	 */
	private normalizeAudio(pcm: Float32Array): Float32Array {
		const peak = this.findPeak(pcm);
		if (peak === 0 || peak >= this.targetPeak) {
			return pcm;
		}

		const gain = this.targetPeak / peak;
		const normalized = new Float32Array(pcm.length);
		for (let i = 0; i < pcm.length; i++) {
			normalized[i] = Math.max(-1, Math.min(1, pcm[i] * gain));
		}
		return normalized;
	}

	/**
	 * Find peak amplitude in audio
	 */
	private findPeak(pcm: Float32Array): number {
		let peak = 0;
		for (let i = 0; i < pcm.length; i++) {
			const abs = Math.abs(pcm[i]);
			if (abs > peak) peak = abs;
		}
		return peak;
	}

	/**
	 * Apply fade-in to the beginning of audio
	 */
	private applyCrossfadeIn(pcm: Float32Array): Float32Array {
		const result = new Float32Array(pcm);
		const fadeSamples = Math.min(this.crossfadeSamples, pcm.length);
		for (let i = 0; i < fadeSamples; i++) {
			result[i] *= i / fadeSamples;
		}
		return result;
	}

	/**
	 * Apply fade-out to the end of audio
	 */
	private applyCrossfadeOut(pcm: Float32Array): Float32Array {
		const result = new Float32Array(pcm);
		const fadeSamples = Math.min(this.crossfadeSamples, pcm.length);
		const startIdx = pcm.length - fadeSamples;
		for (let i = 0; i < fadeSamples; i++) {
			result[startIdx + i] *= 1 - (i / fadeSamples);
		}
		return result;
	}
}

/**
 * Create an AudioContext-based player for streaming playback
 */
export class StreamingAudioPlayer {
	private audioContext: AudioContext | null = null;
	private scheduledTime: number = 0;
	private isPlaying: boolean = false;
	private gainNode: GainNode | null = null;

	/**
	 * Initialize the audio context (must be called from user interaction)
	 */
	async initialize(): Promise<void> {
		if (this.audioContext) return;

		this.audioContext = new AudioContext();
		this.gainNode = this.audioContext.createGain();
		this.gainNode.connect(this.audioContext.destination);
		this.scheduledTime = this.audioContext.currentTime;
	}

	/**
	 * Schedule a chunk for playback
	 */
	async scheduleChunk(chunk: AudioChunk): Promise<void> {
		if (!this.audioContext || !this.gainNode) {
			throw new Error('AudioContext not initialized. Call initialize() first.');
		}

		// Decode the audio
		const arrayBuffer = await chunk.blob.arrayBuffer();
		const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);

		// Create buffer source
		const source = this.audioContext.createBufferSource();
		source.buffer = audioBuffer;
		source.connect(this.gainNode);

		// Schedule playback
		const playTime = Math.max(this.scheduledTime, this.audioContext.currentTime);
		source.start(playTime);

		this.scheduledTime = playTime + audioBuffer.duration;
		this.isPlaying = true;
	}

	/**
	 * Set volume (0-1)
	 */
	setVolume(volume: number): void {
		if (this.gainNode) {
			this.gainNode.gain.value = Math.max(0, Math.min(1, volume));
		}
	}

	/**
	 * Stop playback and reset
	 */
	stop(): void {
		if (this.audioContext) {
			this.audioContext.close();
			this.audioContext = null;
			this.gainNode = null;
		}
		this.isPlaying = false;
		this.scheduledTime = 0;
	}

	/**
	 * Check if audio is currently playing
	 */
	getIsPlaying(): boolean {
		return this.isPlaying;
	}
}

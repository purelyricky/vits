import {
	StreamingConfig,
	AudioChunk,
	StreamMetrics,
	ChunkMetrics,
	StreamingSession,
	StreamEventCallback,
	ChunkCallback,
} from './streaming-types';
import { createTextChunks } from './chunking';
import { AudioBufferManager } from './buffer-manager';
import { HF_BASE, ONNX_BASE, PATH_MAP, WASM_BASE } from './fixtures';
import { readBlob, writeBlob } from './opfs';
import { fetchBlob } from './http';
import { ProgressCallback } from './types';

let piperModule: typeof import('./piper.js');
let ort: typeof import('onnxruntime-web');

// Session management
let currentSession: StreamingSession | null = null;
let sessionCounter = 0;

/**
 * Stream text-to-speech synthesis with incremental audio delivery
 *
 * This is the main streaming API that delivers audio chunks progressively,
 * achieving significantly lower time-to-first-audio (TTFA) compared to
 * batch synthesis.
 *
 * @param config - Streaming configuration
 * @param onChunk - Callback invoked for each audio chunk
 * @param onProgress - Optional callback for model download progress
 * @returns Promise resolving to stream metrics
 *
 * @example
 * ```typescript
 * const metrics = await predictStream(
 *   { text: "Hello world. This is streaming TTS.", voiceId: 'en_US-amy-medium' },
 *   (chunk) => {
 *     // Play or process each chunk as it arrives
 *     const audio = new Audio(URL.createObjectURL(chunk.blob));
 *     audio.play();
 *   }
 * );
 * console.log(`Time to first audio: ${metrics.timeToFirstAudio}ms`);
 * ```
 */
export async function predictStream(
	config: StreamingConfig,
	onChunk: ChunkCallback,
	onProgress?: ProgressCallback
): Promise<StreamMetrics> {
	const streamStart = performance.now();
	const sessionId = `stream-${++sessionCounter}`;

	currentSession = {
		id: sessionId,
		state: 'initializing'
	};

	try {
		// Initialize modules
		piperModule = piperModule ?? (await import('./piper.js'));
		ort = ort ?? (await import('onnxruntime-web'));

		ort.env.allowLocalModels = false;
		ort.env.wasm.numThreads = navigator.hardwareConcurrency;
		ort.env.wasm.wasmPaths = ONNX_BASE;

		// Load model configuration
		const path = PATH_MAP[config.voiceId];
		const modelConfigBlob = await getBlob(`${HF_BASE}/${path}.json`);
		const modelConfig = JSON.parse(await modelConfigBlob.text());

		const sampleRate = modelConfig.audio.sample_rate;
		const noiseScale = modelConfig.inference.noise_scale;
		const lengthScale = modelConfig.inference.length_scale;
		const noiseW = modelConfig.inference.noise_w;

		// Load ONNX model
		const modelBlob = await getBlob(`${HF_BASE}/${path}`, onProgress);
		const session = await ort.InferenceSession.create(await modelBlob.arrayBuffer());

		// Create text chunks
		const chunkSize = config.chunkSize ?? 1;
		const lookahead = config.lookahead ?? 1;
		const textChunks = createTextChunks(config.text, chunkSize, lookahead);

		// Initialize buffer manager
		const bufferManager = new AudioBufferManager({
			sampleRate,
			crossfade: true,
			normalize: config.normalizeAudio ?? true
		});

		currentSession.state = 'streaming';

		const chunkMetrics: ChunkMetrics[] = [];
		let firstChunkTime: number | null = null;

		// Process chunks sequentially for streaming
		for (const textChunk of textChunks) {
			if (currentSession.state === 'cancelled') break;

			const chunkStart = performance.now();

			// Phonemize
			const phonemeStart = performance.now();
			const phonemeIds = await phonemizeText(
				piperModule,
				textChunk.text,
				modelConfig.espeak.voice
			);
			const phonemizeTime = performance.now() - phonemeStart;

			// Run inference
			const inferenceStart = performance.now();
			const pcm = await runInference(
				session,
				ort,
				phonemeIds,
				noiseScale,
				lengthScale,
				noiseW,
				modelConfig.speaker_id_map
			);
			const inferenceTime = performance.now() - inferenceStart;

			const totalTime = performance.now() - chunkStart;
			const timeToFirstByte = performance.now() - streamStart;

			if (firstChunkTime === null) {
				firstChunkTime = timeToFirstByte;
			}

			const metrics: ChunkMetrics = {
				phonemizeTime,
				inferenceTime,
				totalTime,
				timeToFirstByte
			};
			chunkMetrics.push(metrics);

			// Create audio chunk
			const audioChunk = bufferManager.addChunk(pcm, textChunk.text, metrics);
			audioChunk.isLast = textChunk.isLast;

			// Deliver chunk
			onChunk(audioChunk);
		}

		bufferManager.finalize();
		currentSession.state = 'completed';

		// Calculate final metrics
		const totalTime = performance.now() - streamStart;
		const totalAudioDuration = bufferManager.getTotalDuration();

		const streamMetrics: StreamMetrics = {
			timeToFirstAudio: firstChunkTime ?? totalTime,
			totalTime,
			averageChunkLatency: chunkMetrics.reduce((sum, m) => sum + m.totalTime, 0) / chunkMetrics.length,
			totalChunks: chunkMetrics.length,
			totalAudioDuration,
			realTimeFactor: totalAudioDuration / (totalTime / 1000),
			chunkMetrics
		};

		currentSession.metrics = streamMetrics;
		return streamMetrics;

	} catch (error) {
		currentSession.state = 'error';
		currentSession.error = error instanceof Error ? error.message : String(error);
		throw error;
	}
}

/**
 * Stream TTS with event-based API
 *
 * @param config - Streaming configuration
 * @param onEvent - Event callback
 * @param onProgress - Optional progress callback for model download
 */
export async function predictStreamEvents(
	config: StreamingConfig,
	onEvent: StreamEventCallback,
	onProgress?: ProgressCallback
): Promise<void> {
	const sessionId = `stream-${++sessionCounter}`;

	onEvent({ type: 'start', sessionId });

	try {
		const metrics = await predictStream(
			config,
			(chunk) => onEvent({ type: 'chunk', chunk }),
			onProgress
		);
		onEvent({ type: 'complete', metrics });
	} catch (error) {
		onEvent({ type: 'error', error: error instanceof Error ? error.message : String(error) });
	}
}

/**
 * Create an async generator for streaming chunks
 *
 * @example
 * ```typescript
 * for await (const chunk of streamChunks(config)) {
 *   playAudio(chunk.blob);
 * }
 * ```
 */
export async function* streamChunks(
	config: StreamingConfig,
	onProgress?: ProgressCallback
): AsyncGenerator<AudioChunk, StreamMetrics, void> {
	const chunks: AudioChunk[] = [];
	let resolveChunk: ((chunk: AudioChunk | null) => void) | null = null;
	let streamComplete = false;
	let streamMetrics: StreamMetrics | null = null;

	// Start streaming in background
	const streamPromise = predictStream(
		config,
		(chunk) => {
			if (resolveChunk) {
				resolveChunk(chunk);
				resolveChunk = null;
			} else {
				chunks.push(chunk);
			}
		},
		onProgress
	).then((metrics) => {
		streamMetrics = metrics;
		streamComplete = true;
		if (resolveChunk) {
			resolveChunk(null);
		}
	});

	// Yield chunks as they arrive
	while (!streamComplete || chunks.length > 0) {
		if (chunks.length > 0) {
			yield chunks.shift()!;
		} else if (!streamComplete) {
			const chunk = await new Promise<AudioChunk | null>((resolve) => {
				resolveChunk = resolve;
			});
			if (chunk) {
				yield chunk;
			}
		}
	}

	await streamPromise;
	return streamMetrics!;
}

/**
 * Cancel the current streaming session
 */
export function cancelStream(): void {
	if (currentSession && currentSession.state === 'streaming') {
		currentSession.state = 'cancelled';
	}
}

/**
 * Get current session state
 */
export function getStreamingSession(): StreamingSession | null {
	return currentSession;
}

/**
 * Phonemize text using piper-phonemize WASM
 * Creates a new phonemizer instance for each call to ensure proper callback binding
 */
async function phonemizeText(
	piperModule: typeof import('./piper.js'),
	text: string,
	espeakVoice: string
): Promise<number[]> {
	return new Promise(async (resolve, reject) => {
		const input = JSON.stringify([{ text: text.trim() }]);

		try {
			// Create phonemizer with callback set at creation time
			// (Emscripten binds print during initialization, can't change it later)
			const phonemizer = await piperModule.createPiperPhonemize({
				print: (data: string) => {
					try {
						resolve(JSON.parse(data).phoneme_ids);
					} catch (e) {
						reject(new Error(`Failed to parse phoneme output: ${e}`));
					}
				},
				printErr: (message: string) => {
					reject(new Error(message));
				},
				locateFile: (url: string) => {
					if (url.endsWith('.wasm')) return `${WASM_BASE}.wasm`;
					if (url.endsWith('.data')) return `${WASM_BASE}.data`;
					return url;
				}
			});

			phonemizer.callMain([
				'-l', espeakVoice,
				'--input', input,
				'--espeak_data', '/espeak-ng-data'
			]);
		} catch (e) {
			reject(e);
		}
	});
}

/**
 * Run ONNX inference for a chunk
 */
async function runInference(
	session: any,
	ort: typeof import('onnxruntime-web'),
	phonemeIds: number[],
	noiseScale: number,
	lengthScale: number,
	noiseW: number,
	speakerIdMap: Record<string, number>
): Promise<Float32Array> {
	const feeds: Record<string, any> = {
		input: new ort.Tensor('int64', phonemeIds, [1, phonemeIds.length]),
		input_lengths: new ort.Tensor('int64', [phonemeIds.length]),
		scales: new ort.Tensor('float32', [noiseScale, lengthScale, noiseW])
	};

	if (Object.keys(speakerIdMap).length > 0) {
		feeds.sid = new ort.Tensor('int64', [0]);
	}

	const { output: { data: pcm } } = await session.run(feeds);
	return pcm as Float32Array;
}

/**
 * Get blob from OPFS cache or fetch
 */
async function getBlob(url: string, callback?: ProgressCallback): Promise<Blob> {
	let blob: Blob | undefined = await readBlob(url);

	if (!blob) {
		blob = await fetchBlob(url, callback);
		await writeBlob(url, blob);
	}

	return blob;
}

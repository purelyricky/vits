import * as tts from '../src/index';

// DOM Elements
const textInput = document.getElementById('text') as HTMLTextAreaElement;
const voiceSelect = document.getElementById('voice') as HTMLSelectElement;
const streamBtn = document.getElementById('streamBtn') as HTMLButtonElement;
const batchBtn = document.getElementById('batchBtn') as HTMLButtonElement;
const progressBar = document.getElementById('progress') as HTMLDivElement;
const statusDot = document.getElementById('statusDot') as HTMLDivElement;
const statusText = document.getElementById('statusText') as HTMLSpanElement;
const chunksContainer = document.getElementById('chunksContainer') as HTMLDivElement;

// Metric elements
const ttfaEl = document.getElementById('ttfa') as HTMLDivElement;
const totalTimeEl = document.getElementById('totalTime') as HTMLDivElement;
const chunksEl = document.getElementById('chunks') as HTMLDivElement;
const rtfEl = document.getElementById('rtf') as HTMLDivElement;
const streamTTFAEl = document.getElementById('streamTTFA') as HTMLDivElement;
const batchTTFAEl = document.getElementById('batchTTFA') as HTMLDivElement;

// Audio context for streaming playback
let audioContext: AudioContext | null = null;
let scheduledTime = 0;

// Results storage for comparison
let lastStreamTTFA: number | null = null;
let lastBatchTTFA: number | null = null;

function setStatus(text: string, active: boolean = false) {
	statusText.textContent = text;
	statusDot.classList.toggle('active', active);
}

function setProgress(percent: number) {
	progressBar.style.width = `${percent}%`;
}

function clearChunks() {
	chunksContainer.innerHTML = '';
}

function addChunk(chunk: tts.AudioChunk) {
	const chunkEl = document.createElement('div');
	chunkEl.className = 'chunk';
	chunkEl.innerHTML = `
		<div class="chunk-index">${chunk.index + 1}</div>
		<div class="chunk-info">
			<div class="chunk-text">${escapeHtml(chunk.text)}</div>
			<div class="chunk-metrics">
				Phonemize: ${chunk.metrics.phonemizeTime.toFixed(0)}ms |
				Inference: ${chunk.metrics.inferenceTime.toFixed(0)}ms |
				Total: ${chunk.metrics.totalTime.toFixed(0)}ms
			</div>
		</div>
		<button class="chunk-play" data-blob-url="${URL.createObjectURL(chunk.blob)}">
			<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
				<path d="M8 5v14l11-7z"/>
			</svg>
		</button>
	`;

	chunkEl.querySelector('.chunk-play')?.addEventListener('click', (e) => {
		const btn = e.currentTarget as HTMLButtonElement;
		const url = btn.dataset.blobUrl;
		if (url) {
			const audio = new Audio(url);
			audio.play();
		}
	});

	chunksContainer.appendChild(chunkEl);
	chunksContainer.scrollTop = chunksContainer.scrollHeight;
}

function escapeHtml(text: string): string {
	const div = document.createElement('div');
	div.textContent = text;
	return div.innerHTML;
}

function formatMs(ms: number): string {
	if (ms < 1000) {
		return `${ms.toFixed(0)}ms`;
	}
	return `${(ms / 1000).toFixed(2)}s`;
}

async function initAudioContext() {
	if (!audioContext) {
		audioContext = new AudioContext();
	}
	if (audioContext.state === 'suspended') {
		await audioContext.resume();
	}
	scheduledTime = audioContext.currentTime;
}

async function playChunkStreaming(chunk: tts.AudioChunk) {
	if (!audioContext) return;

	const arrayBuffer = await chunk.blob.arrayBuffer();
	const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

	const source = audioContext.createBufferSource();
	source.buffer = audioBuffer;
	source.connect(audioContext.destination);

	const playTime = Math.max(scheduledTime, audioContext.currentTime);
	source.start(playTime);
	scheduledTime = playTime + audioBuffer.duration;
}

async function runStreamingSynthesis() {
	const text = textInput.value.trim();
	const voiceId = voiceSelect.value as tts.VoiceId;

	if (!text) {
		alert('Please enter some text');
		return;
	}

	streamBtn.disabled = true;
	batchBtn.disabled = true;
	clearChunks();
	setStatus('Initializing...', true);
	setProgress(0);

	await initAudioContext();

	let chunkCount = 0;
	const sentences = text.split(/[.!?]+/).filter(s => s.trim()).length;

	try {
		const metrics = await tts.predictStream(
			{
				text,
				voiceId,
				chunkSize: 1,
				lookahead: 1
			},
			async (chunk) => {
				chunkCount++;
				setProgress((chunkCount / sentences) * 100);
				setStatus(`Streaming chunk ${chunkCount}/${sentences}...`, true);

				addChunk(chunk);
				await playChunkStreaming(chunk);

				if (chunk.isFirst) {
					ttfaEl.textContent = formatMs(chunk.metrics.timeToFirstByte);
					lastStreamTTFA = chunk.metrics.timeToFirstByte;
					updateComparison();
				}
			},
			(progress) => {
				if (progress.total > 0) {
					const pct = (progress.loaded / progress.total) * 100;
					setStatus(`Downloading model: ${pct.toFixed(0)}%`, true);
				}
			}
		);

		// Update final metrics
		ttfaEl.textContent = formatMs(metrics.timeToFirstAudio);
		totalTimeEl.textContent = formatMs(metrics.totalTime);
		chunksEl.textContent = String(metrics.totalChunks);
		rtfEl.textContent = metrics.realTimeFactor.toFixed(2) + 'x';

		lastStreamTTFA = metrics.timeToFirstAudio;
		updateComparison();

		setStatus('Streaming complete!', false);
		setProgress(100);

	} catch (error) {
		console.error('Streaming synthesis error:', error);
		setStatus(`Error: ${error}`, false);
	} finally {
		streamBtn.disabled = false;
		batchBtn.disabled = false;
	}
}

async function runBatchSynthesis() {
	const text = textInput.value.trim();
	const voiceId = voiceSelect.value as tts.VoiceId;

	if (!text) {
		alert('Please enter some text');
		return;
	}

	streamBtn.disabled = true;
	batchBtn.disabled = true;
	clearChunks();
	setStatus('Running batch synthesis...', true);
	setProgress(0);

	const startTime = performance.now();

	try {
		const blob = await tts.predict(
			{ text, voiceId },
			(progress) => {
				if (progress.total > 0) {
					const pct = (progress.loaded / progress.total) * 100;
					setProgress(pct);
					setStatus(`Downloading model: ${pct.toFixed(0)}%`, true);
				}
			}
		);

		const totalTime = performance.now() - startTime;

		// For batch, TTFA = total time (no streaming)
		ttfaEl.textContent = formatMs(totalTime);
		totalTimeEl.textContent = formatMs(totalTime);
		chunksEl.textContent = '1';
		rtfEl.textContent = '--';

		lastBatchTTFA = totalTime;
		updateComparison();

		// Play the audio
		const audio = new Audio(URL.createObjectURL(blob));
		audio.play();

		// Add as single chunk
		chunksContainer.innerHTML = `
			<div class="chunk">
				<div class="chunk-index">1</div>
				<div class="chunk-info">
					<div class="chunk-text">${escapeHtml(text.slice(0, 100))}${text.length > 100 ? '...' : ''}</div>
					<div class="chunk-metrics">Batch mode - entire text processed at once</div>
				</div>
				<button class="chunk-play" onclick="this.previousElementSibling.querySelector('audio')?.play()">
					<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
						<path d="M8 5v14l11-7z"/>
					</svg>
				</button>
			</div>
		`;

		setStatus('Batch synthesis complete!', false);
		setProgress(100);

	} catch (error) {
		console.error('Batch synthesis error:', error);
		setStatus(`Error: ${error}`, false);
	} finally {
		streamBtn.disabled = false;
		batchBtn.disabled = false;
	}
}

function updateComparison() {
	if (lastStreamTTFA !== null) {
		streamTTFAEl.textContent = formatMs(lastStreamTTFA);
	}
	if (lastBatchTTFA !== null) {
		batchTTFAEl.textContent = formatMs(lastBatchTTFA);
	}
}

// Event listeners
streamBtn.addEventListener('click', runStreamingSynthesis);
batchBtn.addEventListener('click', runBatchSynthesis);

// Make tts available globally for debugging
(window as any).tts = tts;

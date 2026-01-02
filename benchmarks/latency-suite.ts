/**
 * StreamSpeech Latency Benchmark Suite
 *
 * Comprehensive benchmarks for measuring streaming TTS performance.
 * Run in browser via the benchmark demo page.
 */

import * as tts from '../src/index';

export interface BenchmarkResult {
	name: string;
	textLength: number;
	wordCount: number;
	sentenceCount: number;

	// Streaming metrics
	streaming: {
		timeToFirstAudio: number;
		totalTime: number;
		chunkCount: number;
		avgChunkLatency: number;
		realTimeFactor: number;
		chunkLatencies: number[];
	};

	// Batch metrics
	batch: {
		totalTime: number;
	};

	// Improvement metrics
	improvement: {
		ttfaReduction: number;
		ttfaReductionPercent: number;
	};
}

export interface BenchmarkSuite {
	deviceInfo: {
		userAgent: string;
		hardwareConcurrency: number;
		deviceMemory?: number;
		platform: string;
	};
	voiceId: tts.VoiceId;
	timestamp: string;
	results: BenchmarkResult[];
	summary: {
		avgTTFAReduction: number;
		avgTTFAReductionPercent: number;
		totalTests: number;
	};
}

// Test texts of varying lengths
export const TEST_TEXTS = {
	short: {
		name: 'Short (1 sentence)',
		text: 'The quick brown fox jumps over the lazy dog.'
	},
	medium: {
		name: 'Medium (3 sentences)',
		text: 'The quick brown fox jumps over the lazy dog. This is a test of the streaming text-to-speech system. We expect significant latency improvements over batch processing.'
	},
	long: {
		name: 'Long (5 sentences)',
		text: 'The quick brown fox jumps over the lazy dog. This is a test of the streaming text-to-speech system. We expect significant latency improvements over batch processing. Neural text-to-speech has revolutionized how machines communicate with humans. Browser-based inference democratizes access to these powerful technologies.'
	},
	paragraph: {
		name: 'Paragraph (8 sentences)',
		text: 'The quick brown fox jumps over the lazy dog. This is a test of the streaming text-to-speech system. We expect significant latency improvements over batch processing. Neural text-to-speech has revolutionized how machines communicate with humans. Browser-based inference democratizes access to these powerful technologies. The VITS architecture combines variational inference with adversarial training. This enables high-quality speech synthesis with a single forward pass. Our streaming implementation processes text incrementally for lower perceived latency.'
	},
	article: {
		name: 'Article (12 sentences)',
		text: `The quick brown fox jumps over the lazy dog. This is a test of the streaming text-to-speech system. We expect significant latency improvements over batch processing. Neural text-to-speech has revolutionized how machines communicate with humans. Browser-based inference democratizes access to these powerful technologies. The VITS architecture combines variational inference with adversarial training. This enables high-quality speech synthesis with a single forward pass. Our streaming implementation processes text incrementally for lower perceived latency. Users perceive audio quality as higher when playback begins quickly. This psychological effect is known as the "responsiveness heuristic" in UX research. By reducing time to first audio, we improve both real and perceived quality. The future of browser-based AI is streaming and incremental processing.`
	}
};

/**
 * Run a single benchmark comparing streaming vs batch synthesis
 */
export async function runBenchmark(
	text: string,
	name: string,
	voiceId: tts.VoiceId,
	onProgress?: (stage: string) => void
): Promise<BenchmarkResult> {
	const wordCount = text.split(/\s+/).length;
	const sentenceCount = text.split(/[.!?]+/).filter(s => s.trim()).length;

	onProgress?.('Running streaming synthesis...');

	// Run streaming synthesis
	const chunkLatencies: number[] = [];
	const streamMetrics = await tts.predictStream(
		{ text, voiceId, chunkSize: 1 },
		(chunk) => {
			chunkLatencies.push(chunk.metrics.totalTime);
		}
	);

	onProgress?.('Running batch synthesis...');

	// Run batch synthesis
	const batchStart = performance.now();
	await tts.predict({ text, voiceId });
	const batchTime = performance.now() - batchStart;

	// Calculate improvements
	const ttfaReduction = batchTime - streamMetrics.timeToFirstAudio;
	const ttfaReductionPercent = (ttfaReduction / batchTime) * 100;

	return {
		name,
		textLength: text.length,
		wordCount,
		sentenceCount,
		streaming: {
			timeToFirstAudio: streamMetrics.timeToFirstAudio,
			totalTime: streamMetrics.totalTime,
			chunkCount: streamMetrics.totalChunks,
			avgChunkLatency: streamMetrics.averageChunkLatency,
			realTimeFactor: streamMetrics.realTimeFactor,
			chunkLatencies
		},
		batch: {
			totalTime: batchTime
		},
		improvement: {
			ttfaReduction,
			ttfaReductionPercent
		}
	};
}

/**
 * Run the full benchmark suite
 */
export async function runBenchmarkSuite(
	voiceId: tts.VoiceId = 'en_US-hfc_female-medium',
	onProgress?: (stage: string, progress: number) => void
): Promise<BenchmarkSuite> {
	const results: BenchmarkResult[] = [];
	const tests = Object.entries(TEST_TEXTS);

	for (let i = 0; i < tests.length; i++) {
		const [key, { name, text }] = tests[i];
		onProgress?.(`Testing: ${name}`, (i / tests.length) * 100);

		const result = await runBenchmark(text, name, voiceId, (stage) => {
			onProgress?.(`${name}: ${stage}`, ((i + 0.5) / tests.length) * 100);
		});
		results.push(result);
	}

	// Calculate summary
	const avgTTFAReduction = results.reduce((sum, r) => sum + r.improvement.ttfaReduction, 0) / results.length;
	const avgTTFAReductionPercent = results.reduce((sum, r) => sum + r.improvement.ttfaReductionPercent, 0) / results.length;

	return {
		deviceInfo: {
			userAgent: navigator.userAgent,
			hardwareConcurrency: navigator.hardwareConcurrency,
			deviceMemory: (navigator as any).deviceMemory,
			platform: navigator.platform
		},
		voiceId,
		timestamp: new Date().toISOString(),
		results,
		summary: {
			avgTTFAReduction,
			avgTTFAReductionPercent,
			totalTests: results.length
		}
	};
}

/**
 * Format benchmark results as a markdown table
 */
export function formatResultsMarkdown(suite: BenchmarkSuite): string {
	let md = `# StreamSpeech Benchmark Results\n\n`;
	md += `**Date:** ${suite.timestamp}\n`;
	md += `**Voice:** ${suite.voiceId}\n`;
	md += `**Device:** ${suite.deviceInfo.platform} (${suite.deviceInfo.hardwareConcurrency} cores)\n\n`;

	md += `## Results\n\n`;
	md += `| Test | Words | Sentences | Stream TTFA | Batch Time | Improvement |\n`;
	md += `|------|-------|-----------|-------------|------------|-------------|\n`;

	for (const r of suite.results) {
		md += `| ${r.name} | ${r.wordCount} | ${r.sentenceCount} | ${r.streaming.timeToFirstAudio.toFixed(0)}ms | ${r.batch.totalTime.toFixed(0)}ms | ${r.improvement.ttfaReductionPercent.toFixed(1)}% |\n`;
	}

	md += `\n## Summary\n\n`;
	md += `- **Average TTFA Reduction:** ${suite.summary.avgTTFAReduction.toFixed(0)}ms\n`;
	md += `- **Average Improvement:** ${suite.summary.avgTTFAReductionPercent.toFixed(1)}%\n`;

	return md;
}

/**
 * Export results as JSON for further analysis
 */
export function exportResultsJSON(suite: BenchmarkSuite): string {
	return JSON.stringify(suite, null, 2);
}

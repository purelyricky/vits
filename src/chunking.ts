import { TextChunk } from './streaming-types';

/**
 * Sentence boundary detection patterns
 * Handles common abbreviations, numbers, and edge cases
 */
const ABBREVIATIONS = new Set([
	'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr', 'vs', 'etc', 'inc', 'ltd', 'co',
	'corp', 'st', 'ave', 'blvd', 'rd', 'dept', 'govt', 'univ', 'jan', 'feb', 'mar',
	'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 'mon', 'tue', 'wed',
	'thu', 'fri', 'sat', 'sun', 'no', 'nos', 'vol', 'vols', 'rev', 'est', 'approx',
	'fig', 'figs', 'eg', 'ie', 'cf', 'al', 'et'
]);

/**
 * Detect sentence boundaries in text with high accuracy
 * Handles abbreviations, numbers, quotes, and special cases
 */
export function detectSentenceBoundaries(text: string): number[] {
	const boundaries: number[] = [];
	const sentenceEnders = /[.!?]/;
	const length = text.length;

	for (let i = 0; i < length; i++) {
		const char = text[i];

		if (!sentenceEnders.test(char)) continue;

		// Check if this is a true sentence boundary
		if (isSentenceBoundary(text, i)) {
			boundaries.push(i);
		}
	}

	return boundaries;
}

/**
 * Check if a position is a true sentence boundary
 */
function isSentenceBoundary(text: string, pos: number): boolean {
	const char = text[pos];

	// Must be a sentence-ending punctuation
	if (!/[.!?]/.test(char)) return false;

	// Check for ellipsis (...)
	if (char === '.' && text[pos + 1] === '.' && text[pos + 2] === '.') {
		return false;
	}

	// Check what comes after
	const afterPunc = text.slice(pos + 1, pos + 3);

	// If followed by lowercase letter without space, not a boundary
	if (/^[a-z]/.test(afterPunc)) return false;

	// If followed by digit (like "3.14"), not a boundary
	if (char === '.' && /^\d/.test(afterPunc)) return false;

	// Check for abbreviations before the period
	if (char === '.') {
		const wordBefore = getWordBefore(text, pos);
		if (ABBREVIATIONS.has(wordBefore.toLowerCase())) {
			// But if followed by capital letter after space, it IS a boundary
			const afterSpace = text.slice(pos + 1).match(/^\s+([A-Z])/);
			if (!afterSpace) return false;
		}

		// Check for initials (single capital letter)
		if (/^[A-Z]$/.test(wordBefore)) {
			const afterSpace = text.slice(pos + 1).match(/^\s+([A-Z]\.)/);
			if (afterSpace) return false; // More initials coming
		}
	}

	// Check for quotes - sentence might continue after closing quote
	if (text[pos + 1] === '"' || text[pos + 1] === "'") {
		const afterQuote = text.slice(pos + 2, pos + 4);
		if (/^\s+[a-z]/.test(afterQuote)) return false;
	}

	// If followed by space and capital letter, very likely a boundary
	if (/^\s+[A-Z"]/.test(afterPunc) || /^\s*$/.test(text.slice(pos + 1))) {
		return true;
	}

	// End of text is a boundary
	if (pos === text.length - 1) return true;

	// Default: if followed by whitespace, consider it a boundary
	return /\s/.test(text[pos + 1] || '');
}

/**
 * Get the word immediately before a position
 */
function getWordBefore(text: string, pos: number): string {
	let start = pos - 1;
	while (start >= 0 && /[a-zA-Z]/.test(text[start])) {
		start--;
	}
	return text.slice(start + 1, pos);
}

/**
 * Split text into sentences
 */
export function splitIntoSentences(text: string): string[] {
	const normalized = text.trim().replace(/\s+/g, ' ');
	const boundaries = detectSentenceBoundaries(normalized);

	if (boundaries.length === 0) {
		return [normalized];
	}

	const sentences: string[] = [];
	let start = 0;

	for (const boundary of boundaries) {
		// Include the punctuation in the sentence
		let end = boundary + 1;

		// Include closing quotes if present
		while (end < normalized.length && /["']/.test(normalized[end])) {
			end++;
		}

		const sentence = normalized.slice(start, end).trim();
		if (sentence) {
			sentences.push(sentence);
		}

		start = end;
	}

	// Add any remaining text
	const remaining = normalized.slice(start).trim();
	if (remaining) {
		sentences.push(remaining);
	}

	return sentences;
}

/**
 * Create text chunks for streaming synthesis
 *
 * @param text - The full text to synthesize
 * @param chunkSize - Number of sentences per chunk (default: 1)
 * @param lookahead - Number of sentences to include as context (default: 1)
 * @returns Array of TextChunk objects
 */
export function createTextChunks(
	text: string,
	chunkSize: number = 1,
	lookahead: number = 1
): TextChunk[] {
	const sentences = splitIntoSentences(text);
	const chunks: TextChunk[] = [];

	for (let i = 0; i < sentences.length; i += chunkSize) {
		const chunkSentences = sentences.slice(i, i + chunkSize);
		const chunkText = chunkSentences.join(' ');

		// Get previous context for prosody continuity
		const prevStart = Math.max(0, i - lookahead);
		const previousContext = i > 0
			? sentences.slice(prevStart, i).join(' ')
			: undefined;

		// Get lookahead context for natural phrasing
		const lookaheadEnd = Math.min(sentences.length, i + chunkSize + lookahead);
		const lookaheadContext = i + chunkSize < sentences.length
			? sentences.slice(i + chunkSize, lookaheadEnd).join(' ')
			: undefined;

		chunks.push({
			text: chunkText,
			index: chunks.length,
			isFirst: i === 0,
			isLast: i + chunkSize >= sentences.length,
			previousContext,
			lookaheadContext
		});
	}

	return chunks;
}

/**
 * Estimate the duration of text in seconds (rough approximation)
 * Average speaking rate is ~150 words per minute
 */
export function estimateDuration(text: string): number {
	const words = text.split(/\s+/).length;
	const wordsPerSecond = 150 / 60; // 2.5 words per second
	return words / wordsPerSecond;
}

/**
 * Calculate optimal chunk size based on target latency
 *
 * @param text - Full text to synthesize
 * @param targetLatencyMs - Target time to first audio in ms
 * @param estimatedProcessingRate - Words processed per second
 * @returns Optimal chunk size in sentences
 */
export function calculateOptimalChunkSize(
	text: string,
	targetLatencyMs: number = 300,
	estimatedProcessingRate: number = 50 // words per second
): number {
	const sentences = splitIntoSentences(text);

	if (sentences.length <= 1) return 1;

	// Target: first chunk should process in ~targetLatencyMs
	const targetWords = (targetLatencyMs / 1000) * estimatedProcessingRate;

	// Find chunk size that keeps first chunk under target
	let chunkSize = 1;
	let wordCount = 0;

	for (let i = 0; i < sentences.length; i++) {
		const sentenceWords = sentences[i].split(/\s+/).length;
		if (wordCount + sentenceWords > targetWords && i > 0) break;
		wordCount += sentenceWords;
		chunkSize = i + 1;
	}

	return Math.max(1, chunkSize);
}

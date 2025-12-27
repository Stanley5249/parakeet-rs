//! SRT subtitle conversion utilities

use melops_asr::types::Token;
use srtlib::{Subtitle, Subtitles, Timestamp};

/// Maximum duration for a single subtitle in seconds
const MAX_SUBTITLE_DURATION: f32 = 5.0;

/// Maximum characters per subtitle line
const MAX_CHARS_PER_SUBTITLE: usize = 80;

/// Convert Tokens to SRT Subtitles, grouping by sentences or time windows.
pub fn to_subtitles(tokens: &[Token]) -> Subtitles {
    let segments = group_into_segments(tokens);

    Subtitles::new_from_vec(
        (1..)
            .zip(segments)
            .map(|(i, segment)| {
                Subtitle::new(
                    i,
                    seconds_to_timestamp(segment.start),
                    seconds_to_timestamp(segment.end),
                    segment.text,
                )
            })
            .collect(),
    )
}

/// A subtitle segment with combined text and timing.
struct Segment {
    text: String,
    start: f32,
    end: f32,
}

/// Group tokens into subtitle-friendly segments.
///
/// Segments are split on:
/// - Sentence boundaries (., !, ?)
/// - Maximum duration exceeded
/// - Maximum character count exceeded
fn group_into_segments(tokens: &[Token]) -> Vec<Segment> {
    if tokens.is_empty() {
        return Vec::new();
    }

    let mut segments = Vec::new();
    let mut current_text = String::new();
    let mut current_start: Option<f32> = None;
    let mut current_end: f32 = 0.0;

    for token in tokens {
        let start = current_start.unwrap_or(token.start);
        let duration = token.end - start;
        let new_text_len = current_text.len() + token.text.len();

        // Check if we should start a new segment (but not for punctuation-only tokens)
        let is_punctuation_only = token.text.trim().chars().all(|c| c.is_ascii_punctuation());
        let should_split = !current_text.is_empty()
            && !is_punctuation_only
            && (duration > MAX_SUBTITLE_DURATION || new_text_len > MAX_CHARS_PER_SUBTITLE);

        if should_split {
            // Finish current segment
            segments.push(Segment {
                text: current_text.trim().to_string(),
                start,
                end: current_end,
            });
            current_text = String::new();
            current_start = Some(token.start);
        }

        // Add token to current segment
        current_text.push_str(&token.text);
        if current_start.is_none() {
            current_start = Some(token.start);
        }
        current_end = token.end;

        // Check for sentence boundary (only if we have real content, not just punctuation)
        if is_sentence_end(&token.text) && has_word_content(&current_text) {
            segments.push(Segment {
                text: current_text.trim().to_string(),
                start: current_start.unwrap_or(token.start),
                end: current_end,
            });
            current_text = String::new();
            current_start = None;
        }
    }

    // Don't forget the last segment
    if has_word_content(&current_text)
        && let Some(start) = current_start
    {
        segments.push(Segment {
            text: current_text.trim().to_string(),
            start,
            end: current_end,
        });
    }

    segments
}

/// Check if text contains actual word content (not just punctuation/whitespace).
fn has_word_content(text: &str) -> bool {
    text.chars().any(|c| c.is_alphanumeric())
}

/// Check if a token ends a sentence.
fn is_sentence_end(text: &str) -> bool {
    let trimmed = text.trim();
    trimmed.ends_with('.') || trimmed.ends_with('!') || trimmed.ends_with('?')
}

/// Convert seconds (f32) to SRT Timestamp (hours:minutes:seconds,milliseconds)
fn seconds_to_timestamp(seconds: f32) -> Timestamp {
    let total_ms = (seconds * 1000.0) as u32;
    let ms = (total_ms % 1000) as u16;
    let total_secs = total_ms / 1000;
    let secs = (total_secs % 60) as u8;
    let total_mins = total_secs / 60;
    let mins = (total_mins % 60) as u8;
    let hours = (total_mins / 60) as u8;

    Timestamp::new(hours, mins, secs, ms)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_tokens_into_sentences() {
        let tokens = vec![
            Token {
                text: " Hello".to_string(),
                start: 0.0,
                end: 0.5,
            },
            Token {
                text: " world".to_string(),
                start: 0.5,
                end: 1.0,
            },
            Token {
                text: ".".to_string(),
                start: 1.0,
                end: 1.1,
            },
            Token {
                text: " How".to_string(),
                start: 1.5,
                end: 2.0,
            },
            Token {
                text: " are".to_string(),
                start: 2.0,
                end: 2.5,
            },
            Token {
                text: " you".to_string(),
                start: 2.5,
                end: 3.0,
            },
            Token {
                text: "?".to_string(),
                start: 3.0,
                end: 3.1,
            },
        ];

        let subtitles = to_subtitles(&tokens);
        assert_eq!(subtitles.len(), 2);
    }

    #[test]
    fn handles_empty_tokens() {
        let tokens: Vec<Token> = vec![];
        let subtitles = to_subtitles(&tokens);
        assert_eq!(subtitles.len(), 0);
    }

    #[test]
    fn splits_long_duration() {
        let tokens = vec![
            Token {
                text: " Word".to_string(),
                start: 0.0,
                end: 1.0,
            },
            Token {
                text: " another".to_string(),
                start: 1.0,
                end: 2.0,
            },
            Token {
                text: " more".to_string(),
                start: 6.0,
                end: 7.0,
            },
        ];

        let segments = group_into_segments(&tokens);
        // Should split because duration exceeds MAX_SUBTITLE_DURATION
        assert!(segments.len() >= 2);
    }
}

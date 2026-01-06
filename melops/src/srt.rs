//! SRT subtitle conversion utilities.
//!
//! Converts ASR segments with timestamps into SRT subtitle format.
//!
//! # Tokenization
//!
//! ASR segments contain normalized subword tokens:
//! - Tokens starting with whitespace indicate new words
//! - Tokens without leading whitespace are subword continuations
//! - Leading spaces are stripped from the first segment of each subtitle
//!
//! # Sentence Splitting
//!
//! Subtitles are split on sentence-ending punctuation (. ! ?).
//! If no punctuation is found, all segments form a single subtitle.

use melops_asr::types::Segment;
use srtlib::{Subtitle, Timestamp};

/// Convert Segments to SRT Subtitles, splitting on sentence boundaries.
///
/// Groups segments into subtitles by splitting on sentence-ending punctuation.
/// Strips leading space from the first segment of each subtitle to normalize output.
/// If no sentence-ending punctuation exists, all segments form one subtitle.
pub fn to_subtitles(segments: &[Segment]) -> Vec<Subtitle> {
    let mut subtitles = Vec::new();

    let mut i = 0;

    for j in 1..=segments.len() {
        // Check if we're at the end or if previous segment ends a sentence
        let is_end = j == segments.len();
        let ends_sentence = !is_end && is_sentence_end(&segments[j - 1].text);

        if is_end || ends_sentence {
            let segment_group = &segments[i..j];
            if let Some(subtitle) = merge_segments(segment_group, subtitles.len() + 1) {
                subtitles.push(subtitle);
            }
            i = j;
        }
    }

    subtitles
}

fn merge_segments(segments: &[Segment], index: usize) -> Option<Subtitle> {
    match segments {
        [] => None,
        [s] => {
            let text = match s.text.strip_prefix(" ") {
                Some(t) => t.to_string(),
                None => s.text.clone(),
            };

            Some(Subtitle::new(
                index,
                secs_to_timestamp(s.start),
                secs_to_timestamp(s.end),
                text,
            ))
        }
        [start, .., end] => {
            let text = segments
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    if i == 0
                        && let Some(t) = s.text.strip_prefix(" ")
                    {
                        t
                    } else {
                        s.text.as_str()
                    }
                })
                .collect();

            Some(Subtitle::new(
                index,
                secs_to_timestamp(start.start),
                secs_to_timestamp(end.end),
                text,
            ))
        }
    }
}

/// Check if a segment ends a sentence.
fn is_sentence_end(text: &str) -> bool {
    text.ends_with(['.', '!', '?'])
}

/// Convert seconds to SRT Timestamp
fn secs_to_timestamp(secs: f32) -> Timestamp {
    Timestamp::from_milliseconds((secs * 1000.0) as u32)
}

/// Format subtitles as SRT file content.
///
/// Joins subtitle entries with double newlines as required by SRT format.
pub fn display_subtitles(subtitles: &[Subtitle]) -> String {
    subtitles
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Display preview of subtitles (first and last entries).
///
/// Shows first `head_count` and last `tail_count` subtitles with ellipsis separator.
/// If total subtitles fit within `head_count + tail_count`, displays all.
///
/// # Example
///
/// ```rust,ignore
/// // Show first 2 and last 2 subtitles
/// let preview = preview_subtitles(&subtitles, 2, 2);
/// ```
pub fn preview_subtitles(subtitles: &[Subtitle], head_count: usize, tail_count: usize) -> String {
    let total = subtitles.len();

    // If total fits in head + tail, print all
    if total <= head_count + tail_count {
        display_subtitles(subtitles)
    } else {
        let mut out = Vec::new();

        // Show head
        out.extend(subtitles[0..head_count].iter().map(|s| s.to_string()));

        // Show ellipsis
        out.push("...".to_string());

        // Show tail
        out.extend(
            subtitles[(total - tail_count)..total]
                .iter()
                .map(|s| s.to_string()),
        );

        out.join("\n\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_segments_into_sentences() {
        // Realistic ASR output: subword tokens, leading spaces for new words
        let segments = vec![
            Segment {
                text: " Hello".to_string(),
                start: 0.0,
                end: 0.5,
            },
            Segment {
                text: " world".to_string(),
                start: 0.5,
                end: 1.0,
            },
            Segment {
                text: ".".to_string(),
                start: 1.0,
                end: 1.1,
            },
            Segment {
                text: " How".to_string(),
                start: 1.5,
                end: 2.0,
            },
            Segment {
                text: " are".to_string(),
                start: 2.0,
                end: 2.5,
            },
            Segment {
                text: " you".to_string(),
                start: 2.5,
                end: 3.0,
            },
            Segment {
                text: "?".to_string(),
                start: 3.0,
                end: 3.1,
            },
        ];

        let subtitles = to_subtitles(&segments);

        match &subtitles[..] {
            [first, second] => {
                match first {
                    Subtitle {
                        text,
                        start_time,
                        end_time,
                        ..
                    } if text == "Hello world."
                        && *start_time == Timestamp::from_milliseconds(0)
                        && *end_time == Timestamp::from_milliseconds(1100) => {}
                    _ => panic!("unexpected first subtitle: {:?}", first),
                }
                match second {
                    Subtitle {
                        text,
                        start_time,
                        end_time,
                        ..
                    } if text == "How are you?"
                        && *start_time == Timestamp::from_milliseconds(1500)
                        && *end_time == Timestamp::from_milliseconds(3100) => {}
                    _ => panic!("unexpected second subtitle: {:?}", second),
                }
            }
            _ => panic!("expected 2 subtitles, got {}", subtitles.len()),
        }
    }

    #[test]
    fn handles_empty_segments() {
        let segments: Vec<Segment> = vec![];
        let subtitles = to_subtitles(&segments);
        assert!(matches!(subtitles[..], []));
    }

    #[test]
    fn handles_no_sentence_end() {
        // Incomplete sentence with no ending punctuation
        let segments = vec![
            Segment {
                text: " Word".to_string(),
                start: 0.0,
                end: 1.0,
            },
            Segment {
                text: " another".to_string(),
                start: 1.0,
                end: 2.0,
            },
            Segment {
                text: " more".to_string(),
                start: 2.0,
                end: 3.0,
            },
        ];

        let subtitles = to_subtitles(&segments);
        // No sentence end means final group is still created
        match &subtitles[..] {
            [subtitle] if subtitle.text == "Word another more" => {}
            _ => panic!("unexpected subtitles: {:?}", subtitles),
        }
    }

    #[test]
    fn strips_leading_space_from_first_segment() {
        // Single complete sentence
        let segments = vec![
            Segment {
                text: " The".to_string(),
                start: 0.0,
                end: 0.3,
            },
            Segment {
                text: " quick".to_string(),
                start: 0.3,
                end: 0.6,
            },
            Segment {
                text: " brown".to_string(),
                start: 0.6,
                end: 0.9,
            },
            Segment {
                text: " fox".to_string(),
                start: 0.9,
                end: 1.2,
            },
            Segment {
                text: ".".to_string(),
                start: 1.2,
                end: 1.3,
            },
        ];

        let subtitles = to_subtitles(&segments);

        match &subtitles[..] {
            [subtitle] => match subtitle {
                Subtitle {
                    text,
                    start_time,
                    end_time,
                    ..
                } if text == "The quick brown fox."
                    && *start_time == Timestamp::from_milliseconds(0)
                    && *end_time == Timestamp::from_milliseconds(1300) => {}
                _ => panic!("unexpected subtitle: {:?}", subtitle),
            },
            _ => panic!("expected 1 subtitle, got {}", subtitles.len()),
        }
    }

    #[test]
    fn handles_subword_tokens() {
        // Realistic subword tokenization (e.g., "understanding" -> "under" + "standing")
        let segments = vec![
            Segment {
                text: " I".to_string(),
                start: 0.0,
                end: 0.1,
            },
            Segment {
                text: " am".to_string(),
                start: 0.1,
                end: 0.3,
            },
            Segment {
                text: " under".to_string(),
                start: 0.3,
                end: 0.6,
            },
            Segment {
                text: "standing".to_string(), // No leading space = subword
                start: 0.6,
                end: 0.9,
            },
            Segment {
                text: " this".to_string(),
                start: 0.9,
                end: 1.1,
            },
            Segment {
                text: "!".to_string(),
                start: 1.1,
                end: 1.2,
            },
        ];

        let subtitles = to_subtitles(&segments);

        match &subtitles[..] {
            [subtitle] => match subtitle {
                Subtitle {
                    text,
                    start_time,
                    end_time,
                    ..
                } if text == "I am understanding this!"
                    && *start_time == Timestamp::from_milliseconds(0)
                    && *end_time == Timestamp::from_milliseconds(1200) => {}
                _ => panic!("unexpected subtitle: {:?}", subtitle),
            },
            _ => panic!("expected 1 subtitle, got {}", subtitles.len()),
        }
    }
}

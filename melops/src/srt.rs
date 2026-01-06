//! SRT subtitle conversion utilities.
//!
//! Converts ASR segments with timestamps into SRT subtitle format.
//! Splits subtitles on sentence-ending punctuation (. ! ?).

use melops_asr::types::Segment;
use srtlib::{Subtitle, Timestamp};

/// Convert Segments to SRT Subtitles, splitting on sentence boundaries.
pub fn to_subtitles(segments: &[Segment]) -> Vec<Subtitle> {
    if segments.is_empty() {
        return Vec::new();
    }

    let mut subtitles = Vec::new();
    let mut current_text = String::new();
    let mut current_start: Option<f32> = None;
    let mut current_end: f32 = 0.0;

    for segment in segments {
        current_text.push_str(&segment.text);

        if current_start.is_none() {
            current_start = Some(segment.start);
        }
        current_end = segment.end;

        // Split on sentence-ending punctuation
        if is_sentence_end(&segment.text) {
            if let Some(start) = current_start {
                subtitles.push(Subtitle::new(
                    subtitles.len() + 1,
                    seconds_to_timestamp(start),
                    seconds_to_timestamp(current_end),
                    current_text.trim().to_string(),
                ));
            }
            current_text = String::new();
            current_start = None;
        }
    }

    // Add final subtitle if text remains
    if !current_text.trim().is_empty()
        && let Some(start) = current_start
    {
        subtitles.push(Subtitle::new(
            subtitles.len() + 1,
            seconds_to_timestamp(start),
            seconds_to_timestamp(current_end),
            current_text.trim().to_string(),
        ));
    }

    subtitles
}

/// Check if a segment ends a sentence.
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
        assert_eq!(subtitles.len(), 2);
    }

    #[test]
    fn handles_empty_segments() {
        let segments: Vec<Segment> = vec![];
        let subtitles = to_subtitles(&segments);
        assert_eq!(subtitles.len(), 0);
    }

    #[test]
    fn handles_no_sentence_end() {
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
        assert_eq!(subtitles.len(), 1);
        assert_eq!(subtitles[0].text, "Word another more");
    }
}

//! SRT subtitle conversion utilities.
//!
//! Converts segments with timestamps into SRT subtitle format.

use melops_asr::types::Segment;
use srtlib::{Subtitle, Timestamp};

/// Convert Segments to SRT Subtitles.
pub fn to_subtitles(segments: &[Segment]) -> Vec<Subtitle> {
    segments
        .iter()
        .zip(1..)
        .map(|(s, i)| create_subtitle(s, i))
        .collect()
}

/// Create a subtitle from a segment.
fn create_subtitle(segment: &Segment, index: usize) -> Subtitle {
    Subtitle::new(
        index,
        secs_to_timestamp(segment.start),
        secs_to_timestamp(segment.end),
        segment.text.clone(),
    )
}

/// Convert seconds to SRT Timestamp
fn secs_to_timestamp(secs: f32) -> Timestamp {
    Timestamp::from_milliseconds((secs * 1000.0) as u32)
}

/// Format subtitles as SRT file content.
pub fn display_subtitles(subtitles: &[Subtitle]) -> String {
    subtitles
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Display preview of subtitles (first and last entries).
pub fn preview_subtitles(subtitles: &[Subtitle], head_count: usize, tail_count: usize) -> String {
    let total = subtitles.len();

    if total <= head_count + tail_count {
        display_subtitles(subtitles)
    } else {
        let mut out = Vec::new();
        out.extend(subtitles[0..head_count].iter().map(|s| s.to_string()));
        out.push("...".to_string());
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
    fn converts_segments_to_subtitles() {
        let segments = vec![
            Segment::new("Hello world.", 0.0, 1.1),
            Segment::new("How are you?", 1.5, 3.1),
        ];

        let subtitles = to_subtitles(&segments);

        assert_eq!(subtitles.len(), 2);
        assert_eq!(subtitles[0].text, "Hello world.");
        assert_eq!(subtitles[1].text, "How are you?");
    }

    #[test]
    fn handles_empty_segments() {
        let segments: Vec<Segment> = vec![];
        let subtitles = to_subtitles(&segments);
        assert!(subtitles.is_empty());
    }
}

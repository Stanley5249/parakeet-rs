//! SRT subtitle conversion utilities
use parakeet_rs::TimedToken;
use srtlib::{Subtitle, Subtitles, Timestamp};

/// Convert TimedTokens to SRT Subtitles
pub fn to_subtitles(tokens: &[TimedToken]) -> Subtitles {
    Subtitles::new_from_vec(
        (1..)
            .zip(tokens)
            .map(|(i, token)| {
                Subtitle::new(
                    i,
                    seconds_to_timestamp(token.start),
                    seconds_to_timestamp(token.end),
                    token.text.clone(),
                )
            })
            .collect(),
    )
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

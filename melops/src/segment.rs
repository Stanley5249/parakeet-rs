//! Regroups ASR segments into optimal subtitle segments using dynamic programming

use melops_asr::types::Segment;
use std::f32;

/// Graph node representing a split point position
#[derive(Clone, Copy, Debug)]
struct Node {
    /// Position index (0 = before first segment, n = after last segment)
    index: usize,
    /// Type of split at this position
    split_type: SplitType,
}

/// Split point type between segments
#[derive(Clone, Copy, Debug)]
enum SplitType {
    /// Chunk boundary (start/end)
    Boundary,
    /// After sentence-ending punctuation (., !, ?)
    SentenceEnd,
    /// After soft break punctuation (,, :, ;, -)
    SoftBreak,
    /// At word boundary with silence gap duration
    WordBoundary { gap: f32 },
}

impl Node {
    /// Create a node from segments at position `index` (after `left`, before `right`)
    /// Returns None for invalid mid-word splits
    fn new(index: usize, left: &Segment, right: Option<&Segment>) -> Option<Self> {
        let split_type = if left.text.ends_with(['.', '!', '?']) {
            SplitType::SentenceEnd
        } else if left.text.ends_with([',', ':', ';', '-']) {
            SplitType::SoftBreak
        } else if let Some(right) = right
            && right.text.starts_with(' ')
        {
            let gap = right.start - left.end;
            SplitType::WordBoundary { gap }
        } else {
            return None;
        };

        Some(Self { index, split_type })
    }

    /// Build graph nodes from segments
    ///
    /// Creates nodes at positions between segments:
    ///
    /// | Node | Segments   |
    /// |------|------------|
    /// | 0    | ..0        |
    /// | 1    | 0..1       |
    /// | 2    | 1..2       |
    /// | x    | x-1..x     |
    fn from_segments(segments: &[Segment]) -> Vec<Self> {
        let n = segments.len();

        let mut nodes = vec![Self {
            index: 0,
            split_type: SplitType::Boundary,
        }];

        for i in 1..n {
            let left = &segments[i - 1];
            let right = segments.get(i);

            if let Some(node) = Self::new(i, left, right) {
                nodes.push(node);
            }
        }

        nodes.push(Self {
            index: n,
            split_type: SplitType::Boundary,
        });

        nodes
    }

    /// Base penalty for split point type
    ///
    /// Lower penalty = better split. Scale:
    /// - 0: Perfect (sentence end, boundary)
    /// - 25-40: Good (soft break, long silence)
    /// - 50-100: Poor (short silence)
    ///
    /// Examples:
    /// - "Hello." → "World": 0 (sentence end)
    /// - "However," → "I think": 40 (soft break)
    /// - 1.5s silence: 25 (long pause)
    /// - 0.5s silence: 75 (short pause)
    fn base_penalty(&self) -> f32 {
        match self.split_type {
            SplitType::Boundary => 0.0,
            SplitType::SentenceEnd => 0.0,
            SplitType::SoftBreak => 40.0,
            SplitType::WordBoundary { gap } => {
                if gap >= 1.5 {
                    25.0
                } else {
                    100.0 - gap * 50.0
                }
            }
        }
    }
}

/// Segment regrouping configuration
#[derive(Clone, Copy, Debug)]
pub struct Segmenter {
    /// Maximum silence gap between segments (default: 1.5s)
    /// DP edges with longer gaps are rejected
    pub max_gap: f32,

    /// Target duration (default: 3.0s)
    /// Penalty increases quadratically with distance from target
    pub target_duration: f32,

    /// Target character count (default: 42)
    /// Penalty increases linearly with distance from target
    pub target_chars: f32,

    /// Maximum duration hard limit (default: 7.0s)
    /// DP edges exceeding this are rejected
    pub max_duration: f32,

    /// Maximum character hard limit (default: 84)
    /// DP edges exceeding this are rejected
    pub max_chars: usize,

    /// Target reading speed (default: 22 cps)
    /// Penalty only applied when exceeded (too fast to read)
    pub target_cps: f32,
}

impl Segmenter {
    /// Preset optimized for comfortable reading speed
    ///
    /// Targets ~3 second segments with ~42 characters, matching typical
    /// subtitle display duration and reading comfort. Maximum 7 seconds
    /// prevents overwhelming viewers with long text blocks.
    pub const COMFORTABLE: Self = Self {
        max_gap: 1.5,
        target_duration: 3.0,
        target_chars: 42.0,
        max_duration: 7.0,
        max_chars: 84,
        target_cps: 22.0,
    };

    #[cfg(test)]
    const TEST: Self = Self {
        max_gap: 10.0,
        target_duration: 2.0,
        target_chars: 20.0,
        max_duration: 3.0,
        max_chars: 30,
        target_cps: 22.0,
    };

    /// Regroup segments into optimal subtitle segments using dynamic programming
    pub fn regroup(&self, segments: &[Segment]) -> Vec<Segment> {
        if segments.is_empty() {
            return Vec::new();
        }

        // pre-split at large gaps to avoid checking in DP loop
        let mut chunks = Vec::new();
        let mut i = 0;

        for j in 1..segments.len() {
            let gap = segments[j].start - segments[j - 1].end;
            if gap > self.max_gap {
                chunks.push(i..j);
                i = j;
            }
        }
        chunks.push(i..segments.len());

        chunks
            .into_iter()
            .flat_map(|range| self.regroup_chunk(&segments[range]))
            .collect()
    }

    /// Regroup a chunk using dynamic programming
    fn regroup_chunk(&self, segments: &[Segment]) -> Vec<Segment> {
        if segments.is_empty() {
            return Vec::new();
        }

        let nodes = Node::from_segments(segments);
        let prefix_sum_of_chars = build_char_prefix_sum(segments);

        let (_, parent) = self.find_shortest_path(&nodes, segments, &prefix_sum_of_chars);

        self.backtrack_path(&nodes, &parent, segments)
    }

    /// Find shortest path through nodes using dynamic programming
    fn find_shortest_path(
        &self,
        nodes: &[Node],
        segments: &[Segment],
        prefix_sum_of_chars: &[usize],
    ) -> (Vec<f32>, Vec<Option<usize>>) {
        let n = nodes.len();
        let mut dp = vec![f32::INFINITY; n];
        let mut parent = vec![None; n];
        dp[0] = 0.0;

        for (j, j_node) in nodes.iter().enumerate() {
            for (i, i_node) in nodes.iter().enumerate() {
                if i_node.index >= j_node.index {
                    break;
                }

                let duration = segments[j_node.index - 1].end - segments[i_node.index].start;
                let chars = prefix_sum_of_chars[j_node.index] - prefix_sum_of_chars[i_node.index];

                if duration > self.max_duration || chars > self.max_chars {
                    continue;
                }

                let penalty = j_node.base_penalty() + self.segment_penalty(duration, chars);
                let total = dp[i] + penalty;

                if total < dp[j] {
                    dp[j] = total;
                    parent[j] = Some(i);
                }
            }
        }

        (dp, parent)
    }

    /// Backtrack through parent pointers to build final path
    fn backtrack_path(
        &self,
        nodes: &[Node],
        parent: &[Option<usize>],
        segments: &[Segment],
    ) -> Vec<Segment> {
        let mut path = Vec::new();
        let mut v = nodes.len() - 1;

        while let Some(u) = parent[v] {
            let i = nodes[u].index;
            let j = nodes[v].index;
            v = u;

            if let Some(segment) = merge_segments(&segments[i..j]) {
                path.push(segment);
            }
        }

        path.reverse();
        path
    }

    /// Calculate penalty for segment metrics (duration, chars, reading speed)
    ///
    /// Penalizes deviation from target values. Scale (with defaults):
    /// - Duration: quadratic, ~0-200 range
    ///   - 3.0s (target): 0
    ///   - 2.0s or 4.0s: ~20
    ///   - 1.0s or 5.0s: ~80
    /// - Chars: linear, ~0-20 range
    ///   - 42 chars (target): 0
    ///   - 30 or 54 chars: ~6
    ///   - 20 or 64 chars: ~11
    /// - CPS: exponential when too fast
    ///   - ≤22 cps (target): 0
    ///   - 23 cps: ~8
    ///   - 25 cps: ~32
    fn segment_penalty(&self, duration: f32, chars: usize) -> f32 {
        let mut penalty = 0.0;

        penalty += ((duration - self.target_duration).abs() * 0.45).powi(2);
        penalty += (chars as f32 - self.target_chars).abs() * 0.5;

        let cps = chars as f32 / duration;
        let cps_diff = cps - self.target_cps;
        if cps_diff > 0.0 {
            penalty += (2f32).powf(cps_diff) * 4.0;
        }

        penalty
    }
}

/// Merge consecutive segments into a single segment
fn merge_segments(segments: &[Segment]) -> Option<Segment> {
    match segments {
        [] => None,
        [single] => Some(single.clone()),
        [first, .., last] => {
            let text: String = segments
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    if i == 0 {
                        s.text.strip_prefix(' ').unwrap_or(&s.text)
                    } else {
                        s.text.as_str()
                    }
                })
                .collect();

            Some(Segment::new(text, first.start, last.end))
        }
    }
}

/// Build prefix sum of character counts for fast range queries
fn build_char_prefix_sum(segments: &[Segment]) -> Vec<usize> {
    let mut prefix_sum = vec![0usize; segments.len() + 1];
    for (idx, seg) in segments.iter().enumerate() {
        prefix_sum[idx + 1] = prefix_sum[idx] + seg.text.len();
    }
    prefix_sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regroups_simple_sentence() {
        let segmenter = Segmenter::COMFORTABLE;
        let segments = vec![
            Segment::new(" Hello", 0.0, 0.5),
            Segment::new(" world", 0.5, 1.0),
            Segment::new(".", 1.0, 1.1),
        ];

        let result = segmenter.regroup(&segments);

        match &result[..] {
            [single] => {
                assert_eq!(single.text, "Hello world.");
                assert_eq!(single.start, 0.0);
                assert_eq!(single.end, 1.1);
            }
            _ => panic!("expected 1 segment, got {}", result.len()),
        }
    }

    #[test]
    fn splits_at_large_silence_gap() {
        let segmenter = Segmenter {
            max_gap: 1.0,
            ..Segmenter::COMFORTABLE
        };

        let segments = vec![
            Segment::new(" First", 0.0, 0.5),
            Segment::new(" word", 0.5, 1.0),
            Segment::new(" Second", 2.5, 3.0),
            Segment::new(" word", 3.0, 3.5),
        ];

        let result = segmenter.regroup(&segments);

        assert_eq!(result.len(), 2);
        assert!(result[0].text.contains("First word"));
        assert!(result[1].text.contains("Second word"));
    }

    #[test]
    fn handles_empty_segments() {
        let segmenter = Segmenter::COMFORTABLE;
        let segments: Vec<Segment> = vec![];
        let result = segmenter.regroup(&segments);
        assert!(result.is_empty());
    }

    #[test]
    fn splits_long_sentence() {
        // Total: "This is a very long sentence that needs splitting." = 51 chars, 4.6s duration
        // TEST preset: max 3.0s/30 chars forces split
        let segmenter = Segmenter::TEST;

        let segments = vec![
            Segment::new(" This", 0.0, 0.5),
            Segment::new(" is", 0.5, 1.0),
            Segment::new(" a", 1.0, 1.5),
            Segment::new(" very", 1.5, 2.0),
            Segment::new(" long", 2.0, 2.5),
            Segment::new(" sentence", 2.5, 3.0),
            Segment::new(" that", 3.0, 3.5),
            Segment::new(" needs", 3.5, 4.0),
            Segment::new(" splitting", 4.0, 4.5),
            Segment::new(".", 4.5, 4.6),
        ];

        let result = segmenter.regroup(&segments);

        assert!(
            result.len() >= 2,
            "expected at least 2 segments due to constraint violations"
        );
    }

    #[test]
    fn prefers_silence_gap_split() {
        let segmenter = Segmenter {
            max_gap: 5.0,
            max_duration: 2.5,
            max_chars: 50,
            target_duration: 1.5,
            target_chars: 20.0,
            target_cps: 25.0,
        };

        let segments = vec![
            Segment::new(" First", 0.0, 0.5),
            Segment::new(" part", 0.5, 1.0),
            Segment::new(" second", 2.0, 2.5),
            Segment::new(" part", 2.5, 3.0),
        ];

        let result = segmenter.regroup(&segments);

        match &result[..] {
            [first, second] => {
                assert!(first.text.contains("First part"));
                assert!(second.text.contains("second part"));
            }
            _ => panic!("expected 2 segments, got {}", result.len()),
        }
    }

    #[test]
    fn prefers_sentence_end_over_soft_break() {
        // Force a split with max_chars, choose between sentence end vs soft break
        let segmenter = Segmenter {
            max_chars: 25, // forces split
            max_duration: 100.0,
            max_gap: 100.0,
            target_duration: 100.0,
            target_chars: 1000.0,
            target_cps: 100.0,
        };

        let segments = vec![
            Segment::new(" First part", 0.0, 1.0),  // 11 chars
            Segment::new(".", 1.0, 1.1),            // sentence end: penalty 0
            Segment::new(" Second part", 1.1, 2.0), // 12 chars
            Segment::new(",", 2.0, 2.1),            // soft break: penalty 40
            Segment::new(" Third", 2.1, 3.0),       // 6 chars
        ];
        // Total: 30 chars, must split somewhere
        // Option A: [First part.] + [Second part, Third] = penalty 0 + 40
        // Option B: [First part. Second part,] + [Third] = penalty 40 + boundary
        // Should choose A (split at sentence end)

        let result = segmenter.regroup(&segments);

        assert_eq!(result.len(), 2);
        assert!(result[0].text.contains("First part."));
        assert!(result[1].text.contains("Second part"));
    }

    #[test]
    fn prefers_long_gap_over_short_gap() {
        // Force split, choose between long gap (25) vs short gap (75)
        let segmenter = Segmenter {
            max_chars: 15,
            max_duration: 100.0,
            max_gap: 100.0,
            target_duration: 100.0,
            target_chars: 1000.0,
            target_cps: 100.0,
        };

        let segments = vec![
            Segment::new(" Short", 0.0, 1.0), // 6 chars
            Segment::new(" text", 1.0, 2.0),  // 1.6s gap after: penalty 25
            Segment::new(" more", 3.6, 4.0),  // 5 chars
            Segment::new(" words", 4.0, 4.5), // 0.4s gap after: penalty 80
            Segment::new(" end", 4.9, 5.0),   // 4 chars
        ];

        let result = segmenter.regroup(&segments);

        // Should split at long gap (lower penalty)
        assert!(result.len() >= 2);
        assert!(result[0].text.contains("Short text"));
        assert!(!result[0].text.contains("more"));
    }

    #[test]
    fn keeps_short_sentences_together() {
        let segmenter = Segmenter::COMFORTABLE;

        let segments = vec![
            Segment::new(" Hi", 0.0, 0.3),
            Segment::new("!", 0.3, 0.4),
            Segment::new(" Nice", 0.5, 0.8),
            Segment::new(" to", 0.8, 1.0),
            Segment::new(" meet", 1.0, 1.3),
            Segment::new(" you", 1.3, 1.6),
            Segment::new(".", 1.6, 1.7),
        ];

        let result = segmenter.regroup(&segments);

        match &result[..] {
            [single] => {
                assert_eq!(single.text, "Hi! Nice to meet you.");
            }
            _ => panic!("expected 1 segment, got {}: {:?}", result.len(), result),
        }
    }
}

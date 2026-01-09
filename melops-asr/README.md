# melops-asr

ASR library for NVIDIA Parakeet-TDT models with chunking support for long audio.

## Credits

Complete rewrite of TDT inference pipeline based on [altunenes/parakeet-rs](https://github.com/altunenes/parakeet-rs). Thanks for the insights into TDT architecture and ONNX Runtime setup.

## Key Improvements

- **Decoupled EPs**: Downstream libraries can configure execution providers on their own.
- **Optimization**: Parakeet-TDT pipelines are 2x faster.
- **Chunking**: Handles long audio by splitting into overlapping chunks with smart deduplication.

See root README for usage examples.

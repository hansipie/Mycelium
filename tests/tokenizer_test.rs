use mycelium::model::tokenizer::TokenOutputStream;

// Unit tests for incremental UTF-8 decoding via TokenOutputStream.
// These tests use a fake tokenizer that maps IDs to byte sequences
// to verify partial codepoint handling.

#[test]
fn token_output_stream_yields_complete_ascii() {
    let mut stream = TokenOutputStream::new_test();
    // ASCII characters are always complete codepoints — yield immediately.
    let result = stream.decode_incremental(b"Hello");
    assert_eq!(result, Some("Hello".to_string()));
}

#[test]
fn token_output_stream_buffers_partial_utf8() {
    let mut stream = TokenOutputStream::new_test();
    // First 2 bytes of a 3-byte UTF-8 codepoint (e.g., '€' = 0xE2 0x82 0xAC)
    let result = stream.decode_incremental(&[0xE2, 0x82]);
    assert_eq!(result, None, "incomplete codepoint should not be yielded");
}

#[test]
fn token_output_stream_yields_complete_utf8_when_final_byte_arrives() {
    let mut stream = TokenOutputStream::new_test();
    stream.decode_incremental(&[0xE2, 0x82]); // partial
    let result = stream.decode_incremental(&[0xAC]); // completes '€'
    assert_eq!(result, Some("€".to_string()));
}

#[test]
fn token_output_stream_flush_returns_pending() {
    let mut stream = TokenOutputStream::new_test();
    stream.decode_incremental(&[0xE2]); // partial
    let flushed = stream.flush();
    // A partial sequence that can't be decoded is discarded on flush
    assert!(flushed.is_none() || flushed.unwrap().is_empty());
}

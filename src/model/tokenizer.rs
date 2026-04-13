use tokenizers::Tokenizer;

/// Wraps a HuggingFace tokenizer and decodes tokens incrementally,
/// buffering incomplete UTF-8 codepoints to avoid emitting truncated characters
/// in the SSE stream.
pub struct TokenOutputStream {
    tokenizer: Option<Tokenizer>,
    buffer: Vec<u8>,
}

impl TokenOutputStream {
    pub fn new(tokenizer: Tokenizer) -> Self {
        Self {
            tokenizer: Some(tokenizer),
            buffer: Vec::new(),
        }
    }

    /// Create a test instance without a real tokenizer (for unit tests).
    pub fn new_test() -> Self {
        Self {
            tokenizer: None,
            buffer: Vec::new(),
        }
    }

    /// Decode a token ID and return its text once a complete UTF-8 codepoint is available.
    /// Returns `None` if the decoded bytes don't yet form a complete codepoint.
    pub fn next_token(&mut self, token_id: u32) -> Result<Option<String>, String> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| "no tokenizer".to_string())?;

        let text = tokenizer
            .decode(&[token_id], false)
            .map_err(|e| e.to_string())?;

        Ok(self.decode_incremental(text.as_bytes()))
    }

    /// Decode a raw byte slice incrementally.
    /// Buffers incomplete UTF-8 sequences and returns `Some(text)` when complete.
    pub fn decode_incremental(&mut self, bytes: &[u8]) -> Option<String> {
        self.buffer.extend_from_slice(bytes);
        self.try_flush_buffer()
    }

    fn try_flush_buffer(&mut self) -> Option<String> {
        match std::str::from_utf8(&self.buffer) {
            Ok(s) => {
                let result = s.to_string();
                self.buffer.clear();
                if result.is_empty() {
                    None
                } else {
                    Some(result)
                }
            }
            Err(e) => {
                // Partial UTF-8: keep only the valid prefix and buffer the rest
                let valid_len = e.valid_up_to();
                if valid_len > 0 {
                    let valid = std::str::from_utf8(&self.buffer[..valid_len])
                        .unwrap()
                        .to_string();
                    self.buffer.drain(..valid_len);
                    Some(valid)
                } else {
                    // No complete codepoint yet
                    None
                }
            }
        }
    }

    /// Flush any remaining buffered bytes, discarding incomplete codepoints.
    pub fn flush(&mut self) -> Option<String> {
        if self.buffer.is_empty() {
            return None;
        }
        // Return only the valid UTF-8 prefix; discard any trailing incomplete sequence.
        let result = match std::str::from_utf8(&self.buffer) {
            Ok(s) => s.to_string(),
            Err(e) => std::str::from_utf8(&self.buffer[..e.valid_up_to()])
                .unwrap()
                .to_string(),
        };
        self.buffer.clear();
        if result.is_empty() {
            None
        } else {
            Some(result)
        }
    }
}

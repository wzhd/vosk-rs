# Vosk

[Vosk](https://alphacephei.com/vosk/) is a speech recognition 
toolkit with state-of-the-art accuracy and low latency.
It runs locally and recognises each word as it is uttered without 
waiting for the end of a sentence.

# Usage

Get (or train) a model.
There are trained [models](https://alphacephei.com/vosk/models)
for 16 languages ready for downloading.

Load the model from disk.

```rust
let model = Model::new("path_to_model_directory").unwrap();
```

Create a recognizer object, set the sample rate of audio data.

```rust
let mut recognizer = Recognizer::new(&model, 16000.0);
```

Feed 16-bit audio data as an `i16` slice.

```rust
recognizer.accept_waveform(input);
```

Use `recognizer.partial_result()` to get the words being uttered.
Use `recognizer.result()` to get completed sentences and additional
information such as timing and confidence scores.

# Dependency
Build [vosk-sys](https://github.com/wzhd/vosk-sys).


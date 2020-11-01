use core::fmt;
use serde::{Deserialize, Serialize};
use serde_json::to_writer;
use std::ffi::{CStr, CString};
use std::os::raw::c_int;
use std::path::Path;
use std::sync::Arc;
use vosk_sys::{
    vosk_model_find_word, vosk_model_free, vosk_model_new_or_null,
    vosk_recognizer_accept_waveform_f, vosk_recognizer_accept_waveform_s,
    vosk_recognizer_final_result, vosk_recognizer_free, vosk_recognizer_new,
    vosk_recognizer_new_grm, vosk_recognizer_new_spk, vosk_recognizer_partial_result,
    vosk_recognizer_result, vosk_set_log_level, vosk_spk_model_free, vosk_spk_model_new_or_null,
    VoskModel, VoskRecognizer, VoskSpkModel,
};

/// Stores all the data required for recognition
#[derive(Debug, Clone)]
pub struct Model {
    inner: Arc<ModelInner>,
}

/// Stores all the data required for speaker identification.
#[derive(Debug, Clone)]
pub struct SpeakerModel {
    inner: Arc<SpeakerModelInner>,
}

/// The main object which processes data.
/// Takes audio as input and returns decoded information - words, confidences, times, and so on */
#[derive(Debug)]
pub struct Recognizer {
    ptr: *mut VoskRecognizer,
}

/// The main object which processes data.
/// Takes audio as input and returns decoded information - words, confidences, times, speaker, and so on */
#[derive(Debug)]
pub struct SpeakerRecognizer {
    ptr: *mut VoskRecognizer,
}

#[derive(Debug, PartialEq)]
pub enum Error {
    NoValidModel,
}

/// Set log level for Kaldi messages
///
///   log_level the level
///     0 - default value to print info and error messages but no debug
///     less than 0 - don't print info messages
///     greather than 0 - more verbose mode
pub fn set_log_level(level: c_int) {
    unsafe { vosk_set_log_level(level) }
}

#[derive(Debug)]
struct ModelInner {
    ptr: *mut VoskModel,
}
unsafe impl Sync for ModelInner {}
unsafe impl Send for ModelInner {}

#[derive(Debug)]
struct SpeakerModelInner {
    ptr: *mut VoskSpkModel,
}

unsafe impl Send for SpeakerModelInner {}
unsafe impl Sync for SpeakerModelInner {}

#[derive(Serialize, Deserialize)]
pub struct RecognizedPartial<'a> {
    pub partial: &'a str,
}

/// Speech recognition result
#[derive(Serialize, Deserialize, Debug)]
pub struct RecognizedText<'a> {
    /// May be empty
    pub text: &'a str,
    /// Contains more information about each word when text is not empty
    pub result: Option<Vec<RecognizedWord<'a>>>,
}

/// Information about a word including confidence and timing.
#[derive(Serialize, Deserialize, Debug)]
pub struct RecognizedWord<'a> {
    word: &'a str,
    /// Confidence, less than or equal to 1.0
    conf: f32,
    /// Start time of the word in seconds.
    start: f32,
    end: f32,
}

impl Model {
    // Loads model data from the path
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Model, Error> {
        let path = path_to_cstring(path);
        let model = unsafe { vosk_model_new_or_null(path.as_ptr()) };
        if model.is_null() {
            return Err(Error::NoValidModel);
        }
        let inner = ModelInner { ptr: model };
        let inner = Arc::new(inner);
        Ok(Model { inner })
    }
    /// Check if a word can be recognized by the model
    ///
    /// returns the symbol for `word` if it exists inside the model
    /// or None otherwise.
    /// Note that symbol 0 is for `<epsilon>`
    // Would it be better to return an unsigned number?
    pub fn find_word(&self, word: &str) -> Option<i32> {
        let cstr = CString::new(word).unwrap();
        let sym = unsafe { vosk_model_find_word(self.ptr(), cstr.as_ptr()) };
        if sym == -1 {
            return None;
        }
        Some(sym)
    }
    fn ptr(&self) -> *mut VoskModel {
        self.inner.as_ref().ptr
    }
}

impl SpeakerModel {
    /// Loads speaker model data from the path
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        let path = path_to_cstring(path);
        let model = unsafe { vosk_spk_model_new_or_null(path.as_ptr()) };
        if model.is_null() {
            return Err(Error::NoValidModel);
        }
        let inner = SpeakerModelInner { ptr: model };
        let inner = Arc::new(inner);
        Ok(SpeakerModel { inner })
    }
    fn ptr(&self) -> *mut VoskSpkModel {
        self.inner.as_ref().ptr
    }
}

const INVALID_STR_MSG: &'static str =
    "Invalid UTF-8 in output, which may be from the word list used by the model.";

impl Recognizer {
    /// Creates the recognizer object.
    /// `sample_rate`: The sample rate of the audio that will be fed into the recognizer
    pub fn new(model: &Model, sample_rate: f32) -> Recognizer {
        let recognizer = unsafe { vosk_recognizer_new(model.ptr(), sample_rate) };
        Recognizer { ptr: recognizer }
    }
    ///  Creates the recognizer object with limited subset of words to improve accuracy.
    ///
    /// `word_list` is the list of words separated by spaces.
    ///
    /// Only recognizers with lookahead models support this type of quick configuration.
    ///  Precompiled HCLG graph models are not supported.
    pub fn with_vocabulary(model: &Model, sample_rate: f32, word_list: &str) -> Recognizer {
        Recognizer::with_grammar(
            model,
            sample_rate,
            word_list.split_whitespace().map(|w| Some(w)),
        )
    }
    ///  Creates the recognizer object with limited subset of phrases to improve accuracy.
    ///
    /// `phrases` is anything that can be iterated through and produce each phrase as an item.
    /// A phrase is iterable and produce each word. `Vec<Vec<String>>` works.
    /// Also you can split a `&str` without recollecting.
    ///
    /// ```no_run
    /// # use vosk::{Model, Recognizer};
    /// # let model = Model::new("path_to_model").expect("no model");
    /// let recognizer = Recognizer::with_grammar(
    ///         &model,
    ///         16000.0,
    ///         "link start\nmake tea".lines().map(|p| p.split_whitespace()),
    ///     );
    /// ```
    /// Only recognizers with lookahead models support this type of quick configuration.
    ///  Precompiled HCLG graph models are not supported.
    pub fn with_grammar<I, P, S>(model: &Model, sample_rate: f32, phrases: I) -> Recognizer
    where
        P: IntoIterator<Item = S>,
        I: IntoIterator<Item = P>,
        S: AsRef<str>,
    {
        let mut phrase = String::new();
        let phrase_list: Vec<String> = phrases
            .into_iter()
            .map(|words| {
                phrase.clear();
                words.into_iter().for_each(|s| {
                    let s = s.as_ref();
                    phrase.push_str(s);
                    phrase.push(' ');
                });
                let p = phrase.trim_end();
                p.to_string()
            })
            .collect();
        let mut writer = Vec::with_capacity(128);
        to_writer(&mut writer, &phrase_list).unwrap();
        let cstr = CString::new(writer).unwrap();
        let recognizer =
            unsafe { vosk_recognizer_new_grm(model.ptr(), sample_rate, cstr.as_ptr()) };
        Recognizer { ptr: recognizer }
    }
    /// Accept and process a new chunk of voice data.
    ///
    ///   `data` - audio data in PCM 16-bit mono format.
    ///
    ///  returns true if silence has occurred and you can retrieve a new utterance with `result`,
    ///  otherwise `partial_result` can be used to retrieve an incomplete sentence.
    pub fn accept_waveform(&mut self, wave: &[i16]) -> bool {
        let completed = unsafe {
            vosk_recognizer_accept_waveform_s(self.ptr, wave.as_ptr(), wave.len() as i32)
        };
        completed != 0
    }
    /// Alternative method for processing voice data using f32 instead of i16.
    ///
    ///   `data` - audio data in PCM floating point mono format.
    ///
    ///  returns true if silence has occurred and you can retrieve a new utterance with `result`,
    ///  otherwise `partial_result` can be used to retrieve an incomplete sentence.
    pub fn accept_waveform_f32(&mut self, wave: &[f32]) -> bool {
        let completed = unsafe {
            vosk_recognizer_accept_waveform_f(self.ptr, wave.as_ptr(), wave.len() as i32)
        };
        completed != 0
    }
    /// Returns partial speech recognition text which is not yet finalized,
    /// may change as recognizer processes more data.
    /// Use this when `accept_waveform` returns false.
    pub fn partial_result(&mut self) -> RecognizedPartial {
        let c_str = unsafe {
            let ptr = vosk_recognizer_partial_result(self.ptr);
            CStr::from_ptr(ptr)
        };
        let str = c_str.to_str().expect(INVALID_STR_MSG);
        serde_json::from_str(str).unwrap()
    }
    /// Returns speech recognition result after `accept_waveform` returns true.
    /// Result contains decoded line, decoded words, times in seconds and confidences.
    pub fn result(&mut self) -> RecognizedText {
        let c_str = unsafe {
            let ptr = vosk_recognizer_result(self.ptr);
            CStr::from_ptr(ptr)
        };
        let str = c_str.to_str().expect(INVALID_STR_MSG);
        let r: RecognizedText = serde_json::from_str(str).unwrap();
        r
    }
    /// Returns speech recognition result.
    ///
    ///  Same as `result`, but doesn't wait for silence
    ///  You usually call it in the end of the stream to get final bits of audio. It
    ///  flushes the feature pipeline, so all remaining audio chunks got processed.
    pub fn final_result(&mut self) -> RecognizedText {
        let c_str = unsafe {
            let ptr = vosk_recognizer_final_result(self.ptr);
            CStr::from_ptr(ptr)
        };
        let str = c_str.to_str().expect(INVALID_STR_MSG);
        let r: RecognizedText = serde_json::from_str(str).unwrap();
        r
    }
}

impl SpeakerRecognizer {
    /// Creates the recognizer object with speaker recognition
    ///
    ///  With the speaker recognition mode the recognizer not just recognize
    ///  text but also return speaker vectors one can use for speaker identification
    ///
    ///   `speaker`: speaker model for speaker identification
    ///
    ///   `sample_rate`: The sample rate of the audio you going to feed into the recognizer
    pub fn new(model: &Model, speaker: &SpeakerModel, sample_rate: f32) -> SpeakerRecognizer {
        let recognizer =
            unsafe { vosk_recognizer_new_spk(model.ptr(), speaker.ptr(), sample_rate) };
        SpeakerRecognizer { ptr: recognizer }
    }
}

impl Drop for ModelInner {
    fn drop(&mut self) {
        unsafe {
            vosk_model_free(self.ptr);
        }
    }
}

impl Drop for SpeakerModelInner {
    fn drop(&mut self) {
        unsafe {
            vosk_spk_model_free(self.ptr);
        }
    }
}

impl Drop for Recognizer {
    fn drop(&mut self) {
        unsafe { vosk_recognizer_free(self.ptr) }
    }
}

impl Drop for SpeakerRecognizer {
    fn drop(&mut self) {
        unsafe { vosk_recognizer_free(self.ptr) }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Error::NoValidModel => write!(f, "Could not find valid model at given pat")?,
        }
        Ok(())
    }
}

fn path_to_cstring<P: AsRef<Path>>(path: P) -> CString {
    let path = path.as_ref();
    let path = path_to_bytes(path);
    CString::new(path).unwrap()
}

#[cfg(unix)]
fn path_to_bytes<P: AsRef<Path>>(path: P) -> Vec<u8> {
    use std::os::unix::ffi::OsStrExt;
    path.as_ref().as_os_str().as_bytes().to_vec()
}

#[cfg(not(unix))]
/// Not tested.
fn path_to_bytes<P: AsRef<Path>>(path: P) -> Vec<u8> {
    use std::os::windows::ffi::OsStrExt;
    let path = path.as_ref();
    let mut buf = Vec::new();
    buf.extend(
        path.as_os_str()
            .encode_wide()
            .chain(Some(0))
            .map(|b| {
                let b = b.to_ne_bytes();
                b.get(0).map(|s| *s).into_iter().chain(b.get(1).map(|s| *s))
            })
            .flatten(),
    );
    buf
}

#[cfg(test)]
mod tests {
    use crate::{Error, Model, Recognizer};

    #[test]
    fn not_found() {
        let result = Model::new("not_existing");
        assert_eq!(Error::NoValidModel, result.unwrap_err());
    }
    #[test]
    #[ignore]
    fn one_drop_model() {
        let m = Model::new("model").expect("no model");
        let m1 = m.clone();
        drop(m);
        let _recognizer = Recognizer::new(&m1, 8000.0);
    }
    #[test]
    #[ignore]
    fn word_list() {
        let m = Model::new("model").expect("no model");
        let mut _recognizer = Recognizer::with_vocabulary(&m, 16000.0, "yes no");
    }
    #[test]
    #[ignore]
    fn phrase_list() {
        let m = Model::new("model").expect("no model");
        let v = vec![vec!["hello world"], vec!["initiate the process"]];
        let mut _recognizer = Recognizer::with_grammar(&m, 16000.0, v);
    }
    #[test]
    #[ignore]
    fn share_model() {
        use ::std::thread;
        let m = Model::new("model").expect("no model");
        let m1 = m.clone();
        drop(m);
        thread::spawn(move || {
            let _recognizer = Recognizer::new(&m1, 8000.0);
        })
        .join()
        .unwrap();
    }
}

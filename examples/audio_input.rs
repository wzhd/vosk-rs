use argh::FromArgs;
use portaudio_rs::device::DeviceInfo;
use portaudio_rs::stream::{Stream, StreamCallbackResult, StreamFlags, StreamParameters};
use std::collections::BTreeMap;
use vosk::{Model, Recognizer};

#[derive(FromArgs)]
/// Receive audio and recognize speeches
struct ListenUp {
    /// input device to get audio from.
    #[argh(option, short = 'i')]
    index: Option<u32>,
    /// path to the model
    #[argh(option, short = 'm', default = "String::from(\"model\")")]
    model: String,
    /// number of samples per second
    ///
    #[argh(option, short = 's', default = "default_sample_rate()")]
    sample_rate: f32,
}

fn default_sample_rate() -> f32 {
    16000.0
}

fn main() {
    let devices = list_devices().expect("portaudio failed");
    let up: ListenUp = argh::from_env();
    let i = if let Some(i) = up.index {
        println!("Selected input device {}", i);
        i
    } else {
        let i = portaudio_rs::device::get_default_input_index().expect("no default input");
        println!("Using default input device {}", i);
        i
    };
    let info = devices.get(&i).expect("no device info");

    let model = Model::new(up.model).unwrap();
    let mut recognizer = Recognizer::new(&model, up.sample_rate);
    let mut last_partial = String::new();

    let input_par = StreamParameters {
        device: i,
        channel_count: 1,
        suggested_latency: info.default_low_input_latency,
        data: 42, // random
    };
    let stream = Stream::open(
        Some(input_par),       // input channels
        None,                  // output channels
        up.sample_rate as f64, // sample rate
        portaudio_rs::stream::FRAMES_PER_BUFFER_UNSPECIFIED,
        StreamFlags::empty(),
        Some(Box::new(move |input, _out: &mut [i16], _time, _flags| {
            let completed = recognizer.accept_waveform(input);
            if completed {
                let result = recognizer.final_result();
                if !result.text.is_empty() {
                    println!("{}", result.text);
                }
            } else {
                let result = recognizer.partial_result();
                if result.partial != last_partial {
                    last_partial.clear();
                    last_partial.insert_str(0, &result.partial);
                    if !result.partial.is_empty() {
                        println!("{}", result.partial);
                    }
                }
            }
            StreamCallbackResult::Continue
        })),
    )
    .unwrap();
    stream.start().expect("failed to start the stream");
    std::thread::park();
}

fn list_devices() -> Result<BTreeMap<u32, DeviceInfo>, portaudio_rs::PaError> {
    portaudio_rs::initialize()?;
    let n = portaudio_rs::device::get_count()?;
    let inputs = (0..n)
        .into_iter()
        .filter_map(|index| {
            let info = portaudio_rs::device::get_info(index)?;
            if info.max_input_channels > 0 {
                Some((index, info))
            } else {
                None
            }
        })
        .collect::<BTreeMap<_, _>>();
    if inputs.is_empty() {
        println!("No input devices found.");
    } else {
        println!("Input devices:");
        for (index, info) in inputs.iter() {
            println!("Index={} Name={}", index, info.name);
        }
    }
    Ok(inputs)
}

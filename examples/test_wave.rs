extern crate vosk;

use riff_wave::WaveReader;
use std::fs::File;
use std::io::BufReader;
use vosk::{Model, Recognizer};

fn main() {
    let file = std::env::args()
        .skip(1)
        .next()
        .unwrap_or_else(|| "hello.wav".to_string());
    let file = match File::open(&file) {
        Ok(f) => f,
        Err(e) => {
            println!("Could not open {}: {:?}", file, e);
            return;
        }
    };
    let reader = BufReader::new(file);
    let mut wave_reader = WaveReader::new(reader).expect("wave_reader");
    let fmt = &wave_reader.pcm_format;
    if fmt.num_channels != 1 || fmt.bits_per_sample != 16 {
        println!("Audio file must be WAV format mono PCM.");
        return;
    }
    let mut buf = [0; 1024];
    let model = Model::new("model").unwrap();
    let _recognizer = Recognizer::new(&model, fmt.sample_rate as f32);
    let mut recognizer = Recognizer::with_vocabulary(
        &model,
        fmt.sample_rate as f32,
        "o zero one two three four five six seven eight nine ten",
    );
    let mut last_part = String::new();
    loop {
        let n = read_sample(&mut wave_reader, &mut buf);
        if n == 0 {
            let result = recognizer.final_result();
            println!("Final result: {:?}", result);
            break;
        } else {
            let completed = recognizer.accept_waveform(&buf[..n]);
            if completed {
                let result = recognizer.final_result();
                println!("Result: {:?}", result);
            } else {
                let result = recognizer.partial_result();
                if result.partial != last_part {
                    last_part.clear();
                    last_part.insert_str(0, &result.partial);
                    println!("Partial: {:?}", result.partial);
                }
            }
        }
    }
}

fn read_sample(r: &mut WaveReader<BufReader<File>>, buf: &mut [i16]) -> usize {
    let mut i = 0;
    for _ in 0..buf.len() {
        match r.read_sample_i16() {
            Ok(s) => {
                buf[i] = s;
                i += 1;
            }
            Err(e) => {
                println!("{:?}", e);
                break;
            }
        }
    }
    i
}

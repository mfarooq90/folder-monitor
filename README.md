# Audio File Processor

This repository contains scripts to process audio files using the Whisper model for transcription. There are two main versions:

1. **Folder Watcher Version (`app_dev.py`)**: Monitors a specific directory for new audio files and processes them in real-time.
2. **Directory Processor Version (`process_dir.py`)**: Iterates through an input directory (and its subdirectories) and processes all existing audio files.

Both scripts output transcriptions as `.srt` (subtitle) files and optionally as `.txt` files.

## Features

- Multithreaded processing for faster performance.
- Processes `.mp3`, `.mp4`, and `.wav` files.
- Outputs transcriptions as `.srt` files.
- Optionally outputs transcriptions as `.txt` files.

## 1. Folder Watcher Version (`app_dev.py`)

### Usage

This script continuously monitors a specified directory for new audio files. Once a new file is detected, it's processed using the Whisper model.

### Command Line Arguments

- `input_directory`: The directory to watch and process files from.
- `output_directory`: The directory to save processed files.

### Running the Script

To run the script:

```bash
python app_dev.py [input_directory] [output_directory]
```

Replace `[input_directory]` with the path to the directory you want to watch and `[output_directory]` with the path to the directory where you want to save the processed files.

## 2. Directory Processor Version (`process_dir.py`)

### Usage

This script processes all audio files in a given input directory (and its subdirectories) and saves the processed audio files and their transcriptions in the specified output directory, replicating the original directory structure.

### Command Line Arguments

- `--input_dir`: Path to the input directory containing audio files. Default is `input`.
- `--output_dir`: Path to the output directory where results will be saved. Default is `output`.
- `--max_threads`: Maximum number of threads to use for processing. Default is `10`.
- `--model_name`: Whisper model name to use for processing. Default is `base`.
- `--txt`: Flag to generate transcription as `.txt` file. By default, this is set to `False`.

### Running the Script

To run the script with default values:

```bash
python process_dir.py
```

To specify optional arguments:

```bash
python process_dir.py --input_dir /custom/path/to/input --output_dir /custom/path/to/output --max_threads 5 --model_name large --txt
```

## Dependencies

- `os`
- `shutil`
- `argparse`
- `whisper`
- `datetime`
- `torch`
- `time`
- `concurrent.futures`

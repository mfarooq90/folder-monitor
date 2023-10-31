import os
import shutil
import argparse
import whisper
import datetime
import torch
import time
from concurrent.futures import ThreadPoolExecutor

class FileProcessor:
    def __init__(self, executor, model_name, txt):
        self.executor = executor
        self.model_name = model_name
        self.txt = txt
        self.processed_count = 0  # Initialize counter to 0

    def process_directory(self, dir_path, output_directory):
        for root, dirs, files in os.walk(dir_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                file_extension = os.path.splitext(file_name)[1].lower()

                if file_extension in ['.mp4', '.mp3', '.wav']:
                    self.executor.submit(self.process_file, file_path, root, file_name, dir_path, output_directory)

    def process_file(self, file_path, root, file_name, input_directory, output_directory):
        try:
            start_time = time.time()

            print(f"Processing audio file: {file_name}")
            transcription = None

            if torch.cuda.is_available():
                print("Processing on GPU...")
                model = whisper.load_model(self.model_name, device="cuda")
                transcription = model.transcribe(file_path, device="cuda")
            else:
                model = whisper.load_model(self.model_name, device="cpu")
                transcription = model.transcribe(file_path)

            # Replicate the directory structure in the output directory
            relative_dir = os.path.relpath(root, input_directory)
            output_dir = os.path.join(output_directory, relative_dir)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Save the transcription
            if self.txt:  # Only generate the txt file if the flag is set
                output_file = os.path.join(output_dir, file_name[:-3] + "txt")
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(transcription['text'])


            output_file = os.path.join(output_dir, file_name[:-3] + "srt")
            self.generate_srt(transcription['segments'], output_file)

            # Move the original file to the output directory
            shutil.move(file_path, os.path.join(output_dir, file_name))

            end_time = time.time()
            duration = end_time - start_time
            print(f"Processing of {file_name} completed in {duration:.2f} seconds")

            # After successful processing, increment the counter
            self.processed_count += 1

        except Exception as e:
            print(f"An error occurred while processing {file_name}: {str(e)}")

    def format_srt_time(self, timestamp):
        timestamp = datetime.datetime.utcfromtimestamp(timestamp / 1000.0)
        return timestamp.strftime('%H:%M:%S,%f')[:-3]

    def generate_srt(self, whisper_output, output_srt_filename):
        with open(output_srt_filename, 'w', encoding='utf-8') as srt_file:
            counter = 1
            for item in whisper_output:
                start_time_srt = self.format_srt_time(item['start'])
                end_time_srt = self.format_srt_time(item['end'])
                srt_file.write(f"{counter}\n")
                srt_file.write(f"{start_time_srt} --> {end_time_srt}\n")
                srt_file.write(f"{item['text']}\n")
                srt_file.write("\n")
                counter += 1

def main():
    parser = argparse.ArgumentParser(description="Process audio files in a directory and its subdirectories.")
    parser.add_argument("--input_dir", default="input", help="Path to the input directory containing audio files.")
    parser.add_argument("--output_dir", default="output", help="Path to the output directory where results will be saved.")
    parser.add_argument("--max_threads", default=10, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--model_name", default="base", help="Whisper model name to use for processing.")
    parser.add_argument("--txt", action="store_true", help="Generate transcription as .txt file (default is False).")


    args = parser.parse_args()

    with ThreadPoolExecutor(max_workers=args.max_threads) as executor:
        processor = FileProcessor(executor, args.model_name, args.txt)
        processor.process_directory(args.input_dir, args.output_dir)

        # After processing, print the total number of files processed
    print(f"Total number of files processed: {processor.processed_count}")

if __name__ == "__main__":
    main()

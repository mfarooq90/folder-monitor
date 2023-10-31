import time
import os
import shutil
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import sys
import whisper
import datetime
import torch
from concurrent.futures import ThreadPoolExecutor

# Define the directory to watch
watch_directory = sys.argv[1]
done_directory = sys.argv[2]
text_directory = "txt"
srt_directory = "srt"

# Check if directories exist or create them
if not os.path.exists(watch_directory):
    os.makedirs(watch_directory)
if not os.path.exists(done_directory):
    os.makedirs(done_directory)
if not os.path.exists(text_directory):
    os.makedirs(text_directory)
if not os.path.exists(srt_directory):
    os.makedirs(srt_directory)

class FileHandler(FileSystemEventHandler):
    def __init__(self, executor):
        self.executor = executor

    def on_created(self, event):
        if not event.is_directory:
            file_path = event.src_path
            file_name = os.path.basename(file_path)
            file_extension = os.path.splitext(file_name)[1].lower()

            if file_extension in ['.mp4', '.mp3', '.wav']:
                print(f"Detected audio file: {file_name}")

                # Submit the task to the thread pool
                self.executor.submit(self.process_file, file_path, file_name)

    def process_file(self, file_path, file_name):
        try:
            start_time = time.time()  # Capture the start time

            # Your processing logic remains unchanged
            print(f"Processing audio file: {file_name}")
            transcription = None

            if torch.cuda.is_available():
                print("Processing on GPU...")
                model = whisper.load_model("base", device="cuda")
                transcription = model.transcribe(file_path, device="cuda")
            else:
                model = whisper.load_model("base", device="cpu")
                transcription = model.transcribe(file_path)

            output_file = os.path.join(text_directory, file_name[:-3] + "txt")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(transcription['text'])

            output_file = os.path.join(srt_directory, file_name[:-3] + "srt")
            self.generate_srt(transcription['segments'], output_file)

            shutil.move(file_path, os.path.join(done_directory, file_name))

            end_time = time.time()  # Capture the end time
            duration = end_time - start_time  # Calculate the duration

            print(f"Processing of {file_name} completed in {duration:.2f} seconds")
            print(f"Moved {file_name} to {done_directory}")

        except Exception as e:
            print(f"An error occurred while processing {file_name}: {str(e)}")

    def format_srt_time(self, timestamp):
        """
        Convert a timestamp in milliseconds to an SRT-compatible time format: 'HH:MM:SS,mmm'
        """
        # Convert to a datetime object
        timestamp = datetime.datetime.utcfromtimestamp(timestamp / 1000.0)
        return timestamp.strftime('%H:%M:%S,%f')[:-3]  # The [:-3] omits the last three digits (microseconds)

    def generate_srt(self, whisper_output, output_srt_filename):
        """
        Generate an SRT file based on the output from the Whisper service.

        :param whisper_output: The output from Whisper, assumed to be a list of dictionaries,
                               each containing 'transcript', 'start_time', and 'end_time' keys.
        :param output_srt_filename: The path and filename of the resulting SRT file.
        """
        # Open the output file
        with open(output_srt_filename, 'w', encoding='utf-8') as srt_file:
            counter = 1

            # Process each transcript item
            for item in whisper_output:
                # Get the start and end times in SRT format
                start_time_srt = self.format_srt_time(item['start'])
                end_time_srt = self.format_srt_time(item['end'])

                # Write this subtitle to the file
                srt_file.write(f"{counter}\n")
                srt_file.write(f"{start_time_srt} --> {end_time_srt}\n")
                srt_file.write(f"{item['text']}\n")
                srt_file.write("\n")  # Blank line indicating the start of a new subtitle

                counter += 1

def main():
    max_threads = 10

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        event_handler = FileHandler(executor)
        observer = Observer()
        observer.schedule(event_handler, path=watch_directory, recursive=False)

        observer.start()
        print(f"Monitoring {watch_directory}...")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()

        observer.join()

if __name__ == "__main__":
    main()

import time
import os
import shutil
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import sys
import whisper
import datetime
import torch

# Define the directory to watch
watch_directory = sys.argv[1]
done_directory = sys.argv[2]
text_directory = "txt"
srt_directory = "srt"

# Check if done_directory exists or create it
if not os.path.exists(watch_directory):
    os.makedirs(watch_directory)
if not os.path.exists(done_directory):
    os.makedirs(done_directory)
if not os.path.exists(text_directory):
    os.makedirs(text_directory)
if not os.path.exists(srt_directory):
    os.makedirs(srt_directory)

class FileHandler(FileSystemEventHandler):
    def on_created(self, event):
        # Check if the object created in the directory is a file
        if not event.is_directory:
            file_path = event.src_path
            file_name = os.path.basename(file_path)
            file_extension = os.path.splitext(file_name)[1].lower()  # Get the file extension in lowercase

            # Process only if the file is an mp3 or wav file
            if file_extension in ['.mp4', '.mp3', '.wav']:
                print(f"Processing audio file: {file_name}")
                # Process the file (add your audio processing logic here)
                self.process_file(file_path, file_name)
                
                # Move file to done directory
                shutil.move(file_path, os.path.join(done_directory, file_name))
                print(f"Moved {file_name} to {done_directory}")

    def process_file(self, file_path, filename):
        """
        Define your file processing logic here.
        This is a placeholder function that you should update according to your needs.
        """
        transcription = None
        # Check if CUDA is available
        if torch.cuda.is_available():
            print("Processing on GPU...")
            model = whisper.load_model("large", device="cuda")
            transcription = model.transcribe(file_path, device="cuda")
        else:
            model = whisper.load_model("large", device="cpu")
            transcription = model.transcribe(file_path)


        output_file = os.path.join(text_directory, filename[:-3] + "txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(transcription['text'])

        output_file = os.path.join(srt_directory, filename[:-3] + "srt")
        self.generate_srt(transcription['segments'], output_file)
        # with open(output_file, 'w') as f:
        #     f.write(transcription['segments'])

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
    event_handler = FileHandler()
    observer = Observer()
    observer.schedule(event_handler, path=watch_directory, recursive=False)
    
    observer.start()
    print(f"Monitoring {watch_directory}...")

    try:
        while True:
            # Poll every 1 second.
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()

if __name__ == "__main__":
    main()

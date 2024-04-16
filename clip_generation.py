import os
import json
from clipsai import ClipFinder, Transcriber
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from pytube import YouTube
import whisper
from moviepy.editor import *
##packages for reframing and resizing
from ultralytics import YOLO
import cv2
import subprocess
import numpy as np
import moviepy.editor as mp
from moviepy.editor import VideoFileClip
import os



# Define file paths
json_filename = 'themes/Sigma/video_pull/id.json'
executed_json_filename = 'themes/Sigma/video_pull/executed_id.json'
videos_path = 'themes/Sigma/video_pull/sample/videos'
audio_path = 'themes/Sigma/video_pull/sample/audios'
transcriptions_path = 'themes/Sigma/video_pull/sample/transcripts'
clips_path = 'themes/Sigma/clips'

# Function to download video from YouTube
def download_video(video_url, cleaned_title):
    yt = YouTube(video_url)
    video = yt.streams.get_highest_resolution()
    video.download(videos_path, filename=f"{cleaned_title}.mp4")
    print(f"Downloaded video: {cleaned_title}.mp4")

# Function to download audio from YouTube
def download_audio(video_url, cleaned_title):
    yt = YouTube(video_url)
    audio = yt.streams.filter(only_audio=True).first()
    audio.download(audio_path, filename=f"{cleaned_title}.mp3")
    print(f"Downloaded audio: {cleaned_title}.mp3")

# Function to transcribe audio
def transcribe_audio(audio_filename):
    model = whisper.load_model("tiny.en")
    return model.transcribe(audio_filename, word_timestamps=True, fp16=False)

# Function to save transcription to JSON file
def save_transcription(transcription_result, cleaned_title):
    transcription_filename = os.path.join(transcriptions_path, f"{cleaned_title}.json")
    with open(transcription_filename, 'w') as f:
        json.dump(transcription_result, f, indent=4)
    print(f"Transcription saved to: {transcription_filename}")

# Function to find and process clips
def process_clips(video_filename, audio_filename, cleaned_title):
    transcriber = Transcriber(model_size='tiny')
    transcription = transcriber.transcribe(audio_filename)
    clipfinder = ClipFinder(min_clip_duration=30, max_clip_duration=60)
    clips = clipfinder.find_clips(transcription=transcription)
    clip_number = 1
    for i in clips:
        y = str(clip_number)
        if 30 <= (i.end_time - i.start_time) <= 60:
            clip_filename = os.path.join(clips_path, f"{cleaned_title}{y}temp.mp4")
            clip_filename_resized = os.path.join(clips_path, f"{cleaned_title}{y}resized.mp4")
            final_filename = os.path.join(clips_path, f"{cleaned_title}{y}.mp4")
            ffmpeg_extract_subclip(video_filename, i.start_time, i.end_time, targetname=clip_filename)
            
            clip = mp.VideoFileClip(clip_filename)
            clip_resized = clip.resize(height=1920)
            clip_resized.write_videofile(clip_filename_resized)

            model = YOLO('yolov9c.pt')  # Make sure this points to your model file
            video_path = clip_filename_resized
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print("Error: Could not open video.")
                exit()

            output_width, output_height = 1080, 1920
            output_video_path = 'output_video_corrected.mp4'

            fps = 24
            print("Frames in the video:",int(cap.get(cv2.CAP_PROP_FPS)))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for .mp4 files
            output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))

            # Define a threshold for movement (e.g., 30% of the frame's width)
            movement_threshold = output_width * 0.3
            # Initialize the center position of the last frame for comparison
            last_center_x = None

            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    break

                results = model.predict(frame)
                largest_area = 0
                largest_box = None

                for result in results:
                    for box in result.boxes:
                        b = box.xyxy[0].tolist()
                        area = (b[2] - b[0]) * (b[3] - b[1])
                        if box.cls == 0 and area > largest_area:
                            largest_area = area
                            largest_box = b

                if largest_box:
                    person_center_x = (largest_box[2] + largest_box[0]) / 2
                    
                    # Only adjust the frame if the center has moved significantly
                    if last_center_x is None or abs(person_center_x - last_center_x) > movement_threshold:
                        last_center_x = person_center_x  # Update the last known center
                    else:
                        person_center_x = last_center_x  # Keep the last center if movement is within the threshold
                    
                    start_x = max(0, int(person_center_x - output_width / 2))
                    end_x = min(frame.shape[1], start_x + output_width)
                    
                    # Ensure the cropped area is within the frame's width
                    if end_x - start_x < output_width:
                        start_x = max(0, end_x - output_width)
                    
                    cropped_frame = frame[:, start_x:end_x]
                    
                    # Resize the frame to ensure the height is 1920 pixels
                    resized_frame = cv2.resize(cropped_frame, (output_width, output_height), interpolation=cv2.INTER_LINEAR)
                    
                    output_video.write(resized_frame)

            output_video.release()
            cap.release()
            cv2.destroyAllWindows()

            processed_clip = VideoFileClip(output_video_path)

            # Load the original video to extract the audio
            original_clip = VideoFileClip(clip_filename)

            # Check if the original clip has an audio track
            if original_clip.audio:
                # Set the audio of the processed clip to be the audio of the original clip
                processed_clip = processed_clip.set_audio(original_clip.audio)
                # Write the result to a new file, ensuring the codec is compatible
                processed_clip.write_videofile(final_filename, codec="libx264", audio_codec="aac", fps = fps)
            
                print(f"Finished processing and saved with audio to: {final_filename}")
                os.remove(video_path)
                os.remove(output_video_path)

            else:
                print("The original video does not have an audio track.")
            os.remove(clip_filename)


        clip_number += 1



# Function to process a single video
def process_video(video_url, executed_data):
    try:
        yt = YouTube(video_url)
        video_title = yt.title
        cleaned_title = ''.join(char for char in video_title if char.isalnum() or char in [' ', '.', '_'])

        download_video(video_url, cleaned_title)
        download_audio(video_url, cleaned_title)

        audio_filename = os.path.join(audio_path, f"{cleaned_title}.mp3")
        transcription_result = transcribe_audio(audio_filename)
        save_transcription(transcription_result, cleaned_title)

        video_filename = os.path.join(videos_path, f"{cleaned_title}.mp4")
        process_clips(video_filename, audio_filename, cleaned_title)

        # Add video URL to executed_data
        executed_data.append(video_url)

    except Exception as e:
        print(f"Failed to process video: {video_url}\n{e}")

    return executed_data  # Return the updated executed_data list

def run_clip_generation():
    # Load existing and executed data
    existing_data = []
    executed_data = []

    if os.path.exists(json_filename) and os.path.getsize(json_filename) > 0:
        try:
            with open(json_filename, 'r') as json_file:
                data = json.load(json_file)
                for video_info in data.values():
                    existing_data.append(video_info.get('video_url'))
        except json.decoder.JSONDecodeError:
            print(f"Error: Unable to load JSON data from {json_filename}. File may be empty or corrupted.")

    if os.path.exists(executed_json_filename) and os.path.getsize(executed_json_filename) > 0:
        try:
            with open(executed_json_filename, 'r') as executed_json_file:
                executed_data = json.load(executed_json_file)
        except json.decoder.JSONDecodeError:
            print(f"Error: Unable to load JSON data from {executed_json_filename}. File may be empty or corrupted.")

    # Check which videos need to be processed
    videos_to_process = [video_url for video_url in existing_data if video_url not in executed_data]

    # Process videos that need to be processed
    for video_url in videos_to_process:
        executed_data = process_video(video_url, executed_data)

    # Save updated executed_data to executed_json_filename
    with open(executed_json_filename, 'w') as executed_json_file:
        json.dump(executed_data, executed_json_file, indent=4)
        print("saved video")


if __name__ == "__main__":
    run_clip_generation()


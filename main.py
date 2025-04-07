from typing import List, Dict, Optional, Tuple
from datetime import timedelta
import pyopencl as cl
import whisper
import ffmpeg
import requests
import argparse
import pysrt
import os

def get_video_resolution(video_path:str) -> Optional[Tuple[int, int]]:
	try:
		probe = ffmpeg.probe(video_path)
		video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
		if video_stream:
			width = video_stream.get('width', 0)
			height = video_stream.get('height', 0)
			return width, height
		else:
			return None
	except ffmpeg.Error as e:
		print("Error:", e.stderr.decode())
		return None

def get_subtitles(video_file:str, model:str="small") -> dict:
	model = whisper.load_model(model)
	result = model.transcribe(video_file)
	return result

def translate_subs(text:str, target_lang:str, source_lang:str="auto") -> str:
	return requests.get(f"https://lingva.lunar.icu/api/v1/{source_lang}/{target_lang}/{text}").json()["translation"]

#Convert whisper's output to format required for .srt file.
def seconds_to_srt_time(seconds):
	td = timedelta(seconds=seconds)
	total_seconds = int(td.total_seconds())
	millis = int((td.total_seconds() - total_seconds) * 1000)
	hours = total_seconds // 3600
	minutes = (total_seconds % 3600) // 60
	secs = total_seconds % 60
	return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

#Grab first GPU capable of performing hardware acceleration
def check_gpu() -> Optional[str]:
	platforms = cl.get_platforms()
	for platform in platforms:
		if "NVIDIA" in platform.name:
			return "nvidia"
		elif "AMD" in platform.name:
			return "amd"
	return None

def add_subtitles_to_video(video_file:str, output_file:str, method:str, subtitles:List[Dict], subtitle_fontsize:int=30, subtitle_color:str="white", box_color:str="black", boxborderw:int=5, translate:str=None, gpu_acceleration:bool=False):
	if method == "video":
		x,y = get_video_resolution(video_file)
		if gpu_acceleration != False:
			if (gpu := check_gpu()) == "nvidia":
				video_stream = ffmpeg.input(video_file, hwaccel="cuda").video
			elif gpu == "amd":
				video_stream = ffmpeg.input(video_file, hwaccel="opencl").video
			else:
				print("GPU ACCELERATION NOT SUPPORTED! PROCEEDING WITHOUT.")
				video_stream = ffmpeg.input(video_file).video
		else:
			video_stream = ffmpeg.input(video_file).video
		audio_stream = ffmpeg.input(video_file).audio
		for sub in subtitles["segments"]:
			text = sub["text"]
			if translate != None:
				text = translate_subs(sub["text"], target_lang=translate)
			x_position = (x - (subtitle_fontsize * 0.474 * len(text))) / 2
			video_stream = ffmpeg.drawtext(video_stream, text, x_position, y-70, box=1, boxcolor=box_color, boxborderw=boxborderw, fontcolor=subtitle_color, fontsize=subtitle_fontsize, enable=f"between(t,{sub['start']},{sub['end']})")
		if gpu_acceleration != False:
			if gpu == "nvidia":
				stream = ffmpeg.output(video_stream, audio_stream, output_file, vcodec="h264_nvenc")
			elif gpu == "amd":
				stream = ffmpeg.output(video_stream, audio_stream, output_file, vcodec="h264_amf")
			else:
				stream = ffmpeg.output(video_stream, audio_stream, output_file)
		else:
			stream = ffmpeg.output(video_stream, audio_stream, output_file)
		print(stream.compile())
		stream.run()
		print("Subtitles added. Enjoy!")
	elif method == "file":
		sub_index = 1
		output_file = pysrt.SubRipFile()
		for sub in subtitles["segments"]:
			output_file.append(pysrt.SubRipItem(sub_index, start=seconds_to_srt_time(sub["start"]), end=seconds_to_srt_time(sub["end"]), text=sub["text"]))
			sub_index+=1
		output_name = os.path.splitext(os.path.basename(video_file))[0]+".srt"
		output_file.save(output_name)
		print(f"Subtitle file created for video: {output_name}!")

def main():
	parser = argparse.ArgumentParser(prog="AutoSubtitle", description="Adds subtitles to a video file automatically.")
	parser.add_argument("filename", help="input file")
	parser.add_argument("-o", "--output", help="Output path")
	parser.add_argument("-l", "--language", help="Language of subtitles. (country code)", default=None)
	parser.add_argument("-g", "--gpu", action="store_true", help="GPU Acceleration (Recommended if available)", default=False)
	parser.add_argument("-c", "--color", help="Subtitle color", default="white")
	parser.add_argument("-b", "--boxcolor", help="Subtitle box color", default="black")
	parser.add_argument("-m", "--method", help="'video' = apply sub directly to video, 'file' = create subtitle file.", default="video")
	parser.add_argument("-s", "--size", help="Font size of subtitles", default=30)
	args = parser.parse_args()
	if not os.path.exists(args.filename):
		print("This path does not exist, exiting...")
		return
	if not args.method == "file" and not args.method == "video":
		print("Invalid method, exiting...")
		return
	subs = get_subtitles(args.filename)
	print(args.method)
	add_subtitles_to_video(args.filename, args.output, args.method, subs, gpu_acceleration=args.gpu, translate=args.language, subtitle_color=args.color, box_color=args.boxcolor, subtitle_fontsize=args.size)

if __name__ == "__main__":
	main()
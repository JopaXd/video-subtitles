from typing import List, Dict, Optional, Tuple
import whisper
import ffmpeg
import requests

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

def add_subtitles_to_video(video_file:str, output_file:str, subtitles:List[Dict], subtitle_fontsize:int=30, subtitle_color:str="white", box_color:str="black", font:str="LiberationSans", boxborderw:int=5, translate:str=None):
	x,y = get_video_resolution(video_file)
	video_stream = ffmpeg.input(video_file).video
	audio_stream = ffmpeg.input(video_file).audio
	for sub in subtitles["segments"]:
		text = sub["text"]
		if translate != None:
			text = translate_subs(sub["text"], target_lang=translate)
		x_position = (x - (subtitle_fontsize * 0.474 * len(text))) / 2
		video_stream = ffmpeg.drawtext(video_stream, text, x_position, y-70, box=1, boxcolor=box_color, boxborderw=boxborderw ,fontcolor=subtitle_color, fontsize=subtitle_fontsize, enable=f"between(t,{sub['start']},{sub['end']})")
	stream = ffmpeg.output(video_stream, audio_stream, output_file)
	print(stream.compile())
	stream.run()

def main():
	vid = "vid_30.mp4"
	output = "output.mp4"
	subs = get_subtitles(vid)
	print(subs["segments"])
	add_subtitles_to_video(vid, output, subs, translate="sr")
	print("Subtitles added. Enjoy!")

if __name__ == "__main__":
	main()
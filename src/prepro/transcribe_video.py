import requests
import json
import os
import argparse
import glob
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apikey', type=str, default=None)
    parser.add_argument('--videodir', type=str, default="Data/video/fb")
    parser.add_argument('--outdir', type=str, default="Data/video_transcript/fb")
    parser.add_argument('--model', type=str, default='whisper-1')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    video_files = glob.glob(os.path.join(args.videodir, '*.mp4'))

    for video_file in video_files:
        basename = os.path.basename(video_file)
        _id = basename.replace('.mp4', '')

        outfile = os.path.join(args.outdir, f'{_id}.json')
        if os.path.exists(outfile):
            continue

        headers = {
            "Authorization": f"Bearer {args.apikey}"
        }
        with open(video_file, "rb") as f:
            file_content = f.read()
        files = {
            "file": (basename, file_content, "audio/mp4")
        }
        payload = {
            "name": _id,
            "response_format": "verbose_json",
            "timestamp_granularities": ["segment"],
            "prompt": "transcribe the video",
            "language": "en",
            "model": args.model
        }

        try:
            response = requests.post(
                'https://api.openai.com/v1/audio/transcriptions',
                headers=headers,
                data=payload,
                files=files
            )
            transcript = json.loads(response.content.decode('utf-8'))

            print(_id, transcript)
            if 'text' in transcript:
                if len(transcript['text'].split()) >= 5:
                    with open(outfile, 'w') as f:
                        f.write(json.dumps(transcript, ensure_ascii=False, indent=4))
        except BaseException as e:
            print(f'Something went wrong: {e}')
        time.sleep(5)


if __name__ == '__main__':
    main()

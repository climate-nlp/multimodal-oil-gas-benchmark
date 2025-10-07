# A Multimodal Benchmark for Framing of Oil & Gas Advertising and Potential Greenwashing Detection

This is a codebase to reproduce our work, "A Multimodal Benchmark for Framing of Oil & Gas Advertising and Potential Greenwashing Detection", accepted to NeurIPS 2025 Benchmarks and Datasets Track.

## Setup

#### Recommended Python version:

```
Python 3.12
```

#### Install packages:

```bash
pip install -r requirements.txt
pip install torch==2.6.0 vllm==0.8.3 transformers==4.51.2
```


## Dataset

This project uses Facebook (fb) and YouTube (yt) ad dataset available on [Huggingface datasets](https://huggingface.co/datasets/climate-nlp/multimodal-oil-gas-benchmark).
You have to obtain the videos by your self. We provide video URLs in the dataset to this end. 
This repository does not provide methods for downloading videos, while it is technically easy to implement by yourself. 
For the Facebook ad videos, we only provide the URLs of the ad pages in the dataset. 
To obtain the Facebook ad videos, you might have to locate the HTML tag and class containing the videos on the website and obtain the video URLs.
Some videos may not be available (e.g., deleted by the provider.)

The transcript text can be obtained by using Whisper-1. After obtaining the videos, you can run the following script to transcribe the videos.

```bash
for DATASET in "fb" "yt"
do
python src/prepro/transcribe_video.py \
  --apikey <YOUR OPENAI API KEY> \
  --videodir Data/video/${DATASET} \
  --outdir Data/video_transcript/${DATASET} \
  --model "whisper-1"
done
```

After preparing the dataset, videos, and transcripts, please place the files as follows:
```text
Data/
├── fb_video.train.jsonl
├── fb_video.test.jsonl
├── yt_video.train.jsonl
├── yt_video.test.jsonl
├── video_transcript/fb
├── video_transcript/yt
├── video/fb
└── video/yt
```


## Basic usage

### Prediction by a Vid-LLM
```src/predict.py``` is a script to run a Vid-LLM for inference on our dataset. 
The example usage is:

```bash
# 1-shot with entity restriction (ER) and embedding-based search (ES) for the YouTube domain
python src/predict.py \
    --seed 42 \
    --train_file ./Data/ \
    --test_file ./Data/yt_video.test.jsonl \
    --videodir ./Data/video/yt \
    --transcriptdir ./Data/video_transcript/yt \
    --prompt "yt" \
    --max_frames 10 \
    --outfile predictions.jsonl \
    --model "Qwen/Qwen2.5-VL-7B-Instruct" \
    --num_fewshot 1 \
    --embedding_model "openai/clip-vit-base-patch32" \
    --resized_pixels 65536 \
    --cachedir Cache/

# -> The prediction file will be saved as 'predictions.jsonl'
# To switch the dataset to Facebook, replace 'yt' with 'fb'.
# To disable entity restriction, add the '--no_fewshot_entity_restriction' option.
# To disable embedding-based search, add the '--no_fewshot_embedding_search' option.
```


### Evaluation
```src/score.py``` is a script to evaluate the F-scores of the model output. 
The example usage is:

```bash
python src/score.py \
    -s <your prediction filename>.jsonl \
    -g Data/fb_video.test.jsonl \  # (or Data/yt_video.test.jsonl)
```

Or you can save the evaluation result as a CSV file as follows:

```bash
python src/score.py \
    -s <your prediction filename>.jsonl \
    -g Data/fb_video.test.jsonl \
    -o evaluation_result.csv
```





## Run benchmark experiments for zero-shot and 1-shot prompting

### OpenAI models

```bash
export APIKEY="<YOUR API KEY HERE>"

for DATASET in "yt" "fb"
do
    # Switch the LLM if needed
    #export MODEL_NAME="gpt-4.1-2025-04-14"
    export MODEL_NAME="gpt-4o-mini-2024-07-18"
    
    export MAX_FRAMES=10
    export NUM_FEWSHOT=1
    export TRAIN_FILE="Data/${DATASET}_video.train.jsonl"
    export TEST_FILE="Data/${DATASET}_video.test.jsonl"
    export PROMPT="Data/prompt/${DATASET:0:2}.txt"
    export VIDEO_DIR="Data/video/${DATASET:0:2}"
    export TRANSCRIPT_DIR="Data/video_transcript/${DATASET:0:2}"
    export MODEL=${MODEL_NAME}
    export EMBEDDING_MODEL="openai/clip-vit-base-patch32"
    export OUTDIR=Result/${DATASET}/${MODEL}
    ./scripts/run_openai.sh
done
```

The results will be stored under ```Result/``` directory.

### Qwen2.5-VL

```bash
# 7B
for DATASET in "yt" "fb"
do    
    export MAX_FRAMES=10
    export NUM_FEWSHOT=1
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    export MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
    export MODEL_ABBR="Qwen2.5-VL-7B-Instruct"
    export TRAIN_FILE="Data/${DATASET}_video.train.jsonl"
    export TEST_FILE="Data/${DATASET}_video.test.jsonl"
    export PROMPT="Data/prompt/${DATASET:0:2}.txt"
    export VIDEO_DIR="Data/video/${DATASET:0:2}"
    export TRANSCRIPT_DIR="Data/video_transcript/${DATASET:0:2}"
    export EMBEDDING_MODEL="openai/clip-vit-base-patch32"
    export OUTDIR=Result/${DATASET}/${MODEL_ABBR}
    ./scripts/run_offline.sh
done

# 32B
for DATASET in "yt" "fb"
do
    export MAX_FRAMES=10
    export NUM_FEWSHOT=1
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export MODEL="Qwen/Qwen2.5-VL-32B-Instruct"
    export MODEL_ABBR="Qwen2.5-VL-32B-Instruct"
    export TRAIN_FILE="Data/${DATASET}_video.train.jsonl"
    export TEST_FILE="Data/${DATASET}_video.test.jsonl"
    export PROMPT="Data/prompt/${DATASET:0:2}.txt"
    export VIDEO_DIR="Data/video/${DATASET:0:2}"
    export TRANSCRIPT_DIR="Data/video_transcript/${DATASET:0:2}"
    export EMBEDDING_MODEL="openai/clip-vit-base-patch32"
    export OUTDIR=Result/${DATASET}/${MODEL_ABBR}
    ./scripts/run_offline.sh
done
```


### InternVL2

```bash
# 8B
for DATASET in "yt" "fb"
do
    export MAX_FRAMES=3
    export NUM_FEWSHOT=1
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    export MODEL="OpenGVLab/InternVL2-8B"
    export MODEL_ABBR="InternVL2-8B"
    export TRAIN_FILE="Data/${DATASET}_video.train.jsonl"
    export TEST_FILE="Data/${DATASET}_video.test.jsonl"
    export PROMPT="Data/prompt/${DATASET:0:2}.txt"
    export VIDEO_DIR="Data/video/${DATASET:0:2}"
    export TRANSCRIPT_DIR="Data/video_transcript/${DATASET:0:2}"
    export EMBEDDING_MODEL="openai/clip-vit-base-patch32"
    export OUTDIR=Result/${DATASET}/${MODEL_ABBR}
    ./scripts/run_offline.sh
done
```


### Deepseek VL2

```bash
# 4.5B
for DATASET in "yt" "fb"
do
    export MAX_FRAMES=3
    export NUM_FEWSHOT=1
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    export MODEL="deepseek-ai/deepseek-vl2"
    export MODEL_ABBR="deepseek-vl2"
    export TRAIN_FILE="Data/${DATASET}_video.train.jsonl"
    export TEST_FILE="Data/${DATASET}_video.test.jsonl"
    export PROMPT="Data/prompt/${DATASET:0:2}.txt"
    export VIDEO_DIR="Data/video/${DATASET:0:2}"
    export TRANSCRIPT_DIR="Data/video_transcript/${DATASET:0:2}"
    export EMBEDDING_MODEL="openai/clip-vit-base-patch32"
    export OUTDIR=Result/${DATASET}/${MODEL_ABBR}
    ./scripts/run_offline.sh
done
```



### Experiments for K-shot promoting

```bash
# 7B
for DATASET in "yt" "fb"
do    
    export MAX_FRAMES=10
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    export MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
    export MODEL_ABBR="Qwen2.5-VL-7B-Instruct"
    export TRAIN_FILE="Data/${DATASET}_video.train.jsonl"
    export TEST_FILE="Data/${DATASET}_video.test.jsonl"
    export PROMPT="Data/prompt/${DATASET:0:2}.txt"
    export VIDEO_DIR="Data/video/${DATASET:0:2}"
    export TRANSCRIPT_DIR="Data/video_transcript/${DATASET:0:2}"
    export EMBEDDING_MODEL="openai/clip-vit-base-patch32"
    export OUTDIR=Result/${DATASET}/${MODEL_ABBR}
    ./scripts/run_offline_kshot.sh
done
```



## Citation


```text
@inproceedings{morio-etal-2025-multimodal,
	author = {Morio, Gaku and Rowlands, Harri and Stammbach, Dominik and Manning, Christopher D and Henderson, Peter},
	booktitle = {Advances in Neural Information Processing Systems},
	title = {A Multimodal Benchmark for Framing of Oil \& Gas Advertising and Potential Greenwashing Detection},
	year = {2025}
}
```


## License

See LICENSE.txt under this project.
Note that this project includes third party codes which are not under our license.
Different licenses apply to models and datasets.

We used following libraries:
- [vLLM's inference example](https://github.com/vllm-project/vllm/blob/d1e21a979bba4712f48dac1bbf410e0b57c92e7a/examples/offline_inference_vision_language.py): We adopt the code to ours and made modifications for our task. See ```src/predict.py``` for the original license.

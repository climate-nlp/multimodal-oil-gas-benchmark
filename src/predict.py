# Adapted and modified from: https://github.com/vllm-project/vllm/blob/d1e21a979bba4712f48dac1bbf410e0b57c92e7a/examples/offline_inference_vision_language.py
# The license of the original code is below:
# Apache License
#                            Version 2.0, January 2004
#                         http://www.apache.org/licenses/

"""
This example shows how to use vLLM for running offline inference with
the correct prompt format on vision language models for text generation.

For most models, the prompt format should follow corresponding examples
on HuggingFace model repository.
"""
import json
import os
import random
import pickle

import requests
import argparse
import time
import tqdm
import base64
from io import BytesIO
from typing import List, Optional, Tuple, Dict

import cv2
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoProcessor

from vllm import LLM, SamplingParams
from vllm.multimodal.utils import fetch_image

import qwen_vl_utils


llm: LLM = None


def encode_video(
        video_path: str,
        resized_pixels: int = None,
        sampled_frame_num: int = None,
) -> List[str]:
    # Reference: https://cookbook.openai.com/examples/gpt_with_vision_for_video_understanding
    video = cv2.VideoCapture(video_path)

    frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        if resized_pixels is not None:
            width, height = pil_image.size
            resized_height, resized_width = qwen_vl_utils.vision_process.smart_resize(
                height,
                width,
                factor=1,
                max_pixels=resized_pixels,
            )
            pil_image = pil_image.resize((resized_width, resized_height))

        buffered = BytesIO()
        pil_image.save(buffered, format='JPEG')
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')

        frames.append('data:image/jpeg;base64,' + img_base64)

    video.release()

    # Sample frames
    if sampled_frame_num is not None:
        step = max(int(len(frames) / sampled_frame_num), 1)
        sampled_frames = frames[::step][:sampled_frame_num]
        print(f'Original {len(frames)} frames were sampled to {len(sampled_frames)} frames.')
        return sampled_frames
    else:
        return frames


class VideoData:

    def __init__(self, video_id: str):
        self.video_id: str = video_id
        self.video_path: Optional[str] = None
        self.transcript_path: Optional[str] = None
        self.domain: Optional[str] = None
        self.labels: List[str] = []

        self.max_frames: Optional[int] = None
        self.min_transcript_segment_tokens: Optional[int] = None
        self.resized_pixels: Optional[int] = None
        self.video_length_seconds: Optional[int] = None

        self._frames: List[str] = []  # List of base64 encoded images
        self._transcript_segments: List[Dict] = []
        self._transcript_feature: Optional[torch.tensor] = None
        self._visual_feature: Optional[torch.tensor] = None

    def load_frames_and_transcripts(self):
        if self._frames:
            assert self._transcript_segments
            return

        transcript_segments = []
        if self.transcript_path is not None and os.path.exists(self.transcript_path):
            with open(self.transcript_path, 'r') as f:
                transcript_data = json.loads(f.read())

            for segment in transcript_data['segments']:
                seg_text = segment['text']
                if transcript_data['language'] == 'english' and len(
                        seg_text.split(' ')) < self.min_transcript_segment_tokens:
                    print(
                        f'[{self.transcript_path}] Too few text in the transcript "{seg_text}". We do not consider this.')
                    continue
                if segment['start'] >= self.video_length_seconds or segment['end'] >= self.video_length_seconds:
                    continue
                transcript_segments.append(segment)

        if transcript_segments:
            # Sample frames based on transcript segment time
            all_frames = encode_video(
                self.video_path, resized_pixels=self.resized_pixels, sampled_frame_num=None)
            frames_per_sec = len(all_frames) / self.video_length_seconds
            frames = []
            for segment in transcript_segments:
                mid_seg_time = (segment['start'] + segment['end']) / 2
                target_frame = all_frames[int(mid_seg_time * frames_per_sec + 0.5)]
                frames.append(target_frame)
            frames = frames[:self.max_frames]
            transcript_segments = transcript_segments[:self.max_frames]
        else:
            # Sample frames by constant sampling ratio
            frames = encode_video(
                self.video_path, resized_pixels=self.resized_pixels, sampled_frame_num=self.max_frames)
            transcript_segments = [{'text': 'N/A'} for _ in frames]

        assert len(frames) == len(transcript_segments)
        self._frames = frames
        self._transcript_segments = transcript_segments

    @property
    def frames(self):
        if not self._frames:
            self.load_frames_and_transcripts()
        return self._frames

    @property
    def transcript_segments(self):
        if not self._transcript_segments:
            self.load_frames_and_transcripts()
        return self._transcript_segments

    def load_embeddings(self, model: CLIPModel, processor: CLIPProcessor, tokenizer: AutoTokenizer):
        if self._transcript_feature is not None:
            assert self._visual_feature
            return

        images = map(lambda f: Image.open(BytesIO(base64.b64decode(f.split('base64,')[1]))), self.frames)
        images = list(images)

        texts = map(lambda s: s['text'], self.transcript_segments)
        texts = list(texts)

        text_inputs = tokenizer(texts, padding=True, return_tensors="pt", truncation=True)
        text_features = model.get_text_features(**text_inputs)  # (n_texts, n_dim)
        self._transcript_feature = text_features.mean(0)  # (n_dim,)

        image_inputs = processor(images=images, return_tensors="pt")
        image_features = model.get_image_features(**image_inputs)  # (n_images, n_dim)
        self._visual_feature = image_features.mean(0)  # (n_dim,)

    @property
    def transcript_feature(self):
        return self._transcript_feature

    @property
    def visual_feature(self):
        return self._visual_feature

    def save(self, fpath: str):
        with open(fpath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_from_path(cls, fpath: str):
        with open(fpath, 'rb') as f:
            inst = pickle.load(f)
        return inst


def parse_label(generated_answer: str, label_list: List[str]):
    pred_labels = [l for l in label_list if l in generated_answer]
    pred_labels = sorted(pred_labels)
    return pred_labels


def load_dataset(
        fpath: str, video_dir: str, transcript_dir: str, max_frames: int, resized_pixels: int,
        min_transcript_segment_tokens: int, cache_dir: str or None
) -> List[VideoData]:
    dataset = []
    print(f'Loading dataset from {fpath}. (transcript_dir={transcript_dir}, transcript_dir={transcript_dir}, '
          f'max_frames={max_frames}, resized_pixels={resized_pixels}, '
          f'min_transcript_segment_tokens={min_transcript_segment_tokens})')

    with open(fpath, 'r') as f:
        for l in tqdm.tqdm(f.readlines()):
            d = json.loads(l)

            if cache_dir is not None:
                cache_file = os.path.join(cache_dir, d['video_id'] + '.pkl')
                if os.path.exists(cache_file):
                    dataset.append(
                        VideoData.load_from_path(cache_file)
                    )
                    continue

            vd = VideoData(video_id=d['video_id'])
            video_path = os.path.join(video_dir, d['video_id'] + '.mp4')
            vd.video_path = video_path

            vd.video_length_seconds = d['video_length_seconds']
            vd.max_frames = max_frames
            vd.min_transcript_segment_tokens = min_transcript_segment_tokens
            vd.resized_pixels = resized_pixels

            transcript_path = os.path.join(transcript_dir, d["video_id"] + '.json')
            if os.path.exists(transcript_path):
                vd.transcript_path = transcript_path

            vd.domain = d['entity_name']
            vd.labels = d['labels']
            dataset.append(vd)

    return dataset


def load_existing_output(output_fpath: str) -> Tuple[List[dict], List[str]]:
    existing_output_data = []
    existing_output_data_ids = []
    if os.path.exists(output_fpath):
        with open(output_fpath, 'r') as f:
            existing_output_data += [json.loads(l) for l in f.readlines()]
        existing_output_data_ids = list(map(lambda x: x['video_id'], existing_output_data))
    return existing_output_data, existing_output_data_ids


def video_similarity_search(
    target_data: VideoData, candidate_data: List[VideoData],
    model: CLIPModel, processor: CLIPProcessor, tokenizer: AutoTokenizer,
    topk: int,
    transcript_factor: float = 0.5, visual_factor: float = 0.5,
) -> Tuple[List[VideoData], List[float or None]]:
    # Add embeddings if not exists
    if target_data.visual_feature is None or target_data.transcript_feature is None:
        target_data.load_embeddings(model=model, processor=processor, tokenizer=tokenizer)
    for data in candidate_data:
        if data.visual_feature is None or data.transcript_feature is None:
            data.load_embeddings(model=model, processor=processor, tokenizer=tokenizer)

    def get_feature_fusion(_data: List[VideoData]):
        fusion_featured = [
            (transcript_factor * _d.transcript_feature) + (visual_factor * _d.visual_feature)
            for _d in _data
        ]
        return torch.stack(fusion_featured)

    similarity = torch.nn.functional.cosine_similarity(
        get_feature_fusion([target_data]),
        get_feature_fusion(candidate_data)
    )  # (n_candidates,)

    results = []
    similarities = []
    for ind in torch.topk(similarity, k=min(similarity.shape[0], topk)).indices:
        data = candidate_data[ind]
        similarities.append(similarity[ind].tolist())
        results.append(data)

    results = results[::-1]
    similarities = similarities[::-1]

    return results, similarities


def get_fewshot_data(
        num_fewshot,
        test_data: VideoData,
        train_data_list: List[VideoData],
        default_fewshot_data_list: List[VideoData],
        default_entity_fewshot_data_list: Dict[str, List[VideoData]],
        model: CLIPModel, processor: CLIPProcessor, tokenizer: AutoTokenizer,
        no_fewshot_entity_restriction: bool = False,
        no_fewshot_embedding_search: bool = False,
        embed_transcript_factor: float = 0.5,
        embed_visual_factor: float = 0.5,
) -> Tuple[List[VideoData], List[float or None]]:
    similarities = None
    print(f'[{test_data.video_id}] Selecting few-shot samples.')
    if no_fewshot_entity_restriction:
        print(f'[{test_data.video_id}] Entity restriction is disabled.')
        # No entity restriction
        if no_fewshot_embedding_search:
            print(f'[{test_data.video_id}] Randomly sampled data from train data is used because '
                  'you disabled the "entity restriction" and the "embedding search".')
            final_fewshot_data_list = default_fewshot_data_list
        else:
            print(f'[{test_data.video_id}] Searched data from train data is used because '
                  f'you disabled the "entity restriction".')
            final_fewshot_data_list, similarities = video_similarity_search(
                target_data=test_data, candidate_data=train_data_list,
                model=model, processor=processor, tokenizer=tokenizer,
                topk=num_fewshot,
                transcript_factor=embed_transcript_factor, visual_factor=embed_visual_factor
            )
    else:
        print(f'[{test_data.video_id}] Entity restriction is enabled.')
        domain_train_data = [d for d in train_data_list if d.domain == test_data.domain]
        if not domain_train_data:
            print(f'[{test_data.video_id}] We could not find in-domain(entity) train data for {test_data.domain}.')
            if no_fewshot_embedding_search:
                print(f'[{test_data.video_id}] Instead, we will use randomly sampled few-shot data')
                final_fewshot_data_list = default_fewshot_data_list
            else:
                print(f'[{test_data.video_id}] Instead, we will use searched data from train data.')
                final_fewshot_data_list, similarities = video_similarity_search(
                    target_data=test_data, candidate_data=train_data_list,
                    model=model, processor=processor, tokenizer=tokenizer,
                    topk=num_fewshot,
                    transcript_factor=embed_transcript_factor, visual_factor=embed_visual_factor
                )
        else:
            print(f'[{test_data.video_id}] Found {len(domain_train_data)} in-domain(entity) train data for {test_data.domain}.')
            if no_fewshot_embedding_search:
                if test_data.domain in default_entity_fewshot_data_list:
                    print(f'[{test_data.video_id}] We use randomly sampled in-domain(entity) train data.')
                    final_fewshot_data_list = default_entity_fewshot_data_list[test_data.domain]
                else:
                    print(f'[{test_data.video_id}] We could not find in-domain(entity) train data.'
                          f'Instead, we will use randomly sampled few-shot data')
                    final_fewshot_data_list = default_fewshot_data_list
            else:
                print(f'[{test_data.video_id}] We use searched data from in-domain(entity) train data.')
                final_fewshot_data_list, similarities = video_similarity_search(
                    target_data=test_data, candidate_data=domain_train_data,
                    model=model, processor=processor, tokenizer=tokenizer,
                    topk=num_fewshot,
                    transcript_factor=embed_transcript_factor, visual_factor=embed_visual_factor
                )

    assert final_fewshot_data_list

    if not similarities:
        return final_fewshot_data_list, [None for _ in final_fewshot_data_list]
    else:
        return final_fewshot_data_list, similarities


def run_openai(content: List[dict], apikey: str, model: str, max_retry: int = 10, retry_wait_sec: int = 30) -> str:
    messages = [{
        "role": "user",
        "content": content
    }]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {apikey}"
    }

    if model.startswith('gpt-4'):
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0,
        }
    elif model.startswith('gpt-5'):
        payload = {
            "model": model,
            "messages": messages,
        }
    elif model.startswith('o'):
        payload = {
            "model": model,
            "messages": messages,
        }
    else:
        assert False

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        print(response.json())
        generated_answer = response.json()['choices'][0]['message']['content'].strip()
    except BaseException as e:
        # Retry
        print(e)
        if max_retry > 0:
            print(f'Request error occurred. '
                  f'Retry the same request with a sleep of {retry_wait_sec} sec. '
                  f'{max_retry} retries remaining.')
            time.sleep(retry_wait_sec)
            return run_openai(
                content=content,
                apikey=apikey,
                model=model,
                max_retry=max_retry - 1,
                retry_wait_sec=retry_wait_sec,
            )
        else:
            assert False
    return generated_answer


def run_qwen2p5_vl(content: List[dict], model: str) -> str:
    global llm

    image_urls = [c['image_url']['url'] for c in content if c['type'] == 'image_url']

    if not llm:
        llm = LLM(
            model=model,
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
            #max_model_len=2048,
            enforce_eager=True,
            tensor_parallel_size=torch.cuda.device_count(),
            quantization="AWQ" if "awq" in model.lower() else None,
            dtype="float16" if "awq" in model.lower() else "auto",
            max_num_seqs=1,
            seed=42,
            limit_mm_per_prompt={"image": 15*10},
        )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=32,
    )

    new_content = []
    for i, c in enumerate(content):
        if i == 0:
            assert c['type'] == 'text'
        if c['type'] == 'image_url':
            new_content.append(
                {"type": "image", "image": c['image_url']['url'], "min_pixels": 128*28*28, "max_pixels": 256*28*28}
            )
        else:
            assert c['type'] == 'text'
            new_content.append(c)
    assert len(content) == len(new_content)

    messages = [{
        "role": "user",
        "content": new_content
    }]

    processor = AutoProcessor.from_pretrained(model)
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs, video_kwargs = qwen_vl_utils.process_vision_info(messages, return_video_kwargs=True)
    assert len(image_inputs) == len(image_urls), f'image_inputs: {len(image_inputs)} != image_urls: {len(image_urls)}'

    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    print('#### FINAL PROMPT ####')
    print(prompt)
    print('#######################')

    inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs,
    }

    outputs = llm.generate(inputs, sampling_params=sampling_params)
    generated_answer = outputs[0].outputs[0].text
    return generated_answer


def run_deepseek_vl2(content: List[dict], model: str) -> str:
    global llm

    image_urls = [c['image_url']['url'] for c in content if c['type'] == 'image_url']

    if not llm:
        llm = LLM(
            model=model,
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
            #max_model_len=4096,
            enforce_eager=True,
            tensor_parallel_size=torch.cuda.device_count(),
            hf_overrides={"architectures": ["DeepseekVLV2ForCausalLM"]},
            max_num_seqs=1,
            limit_mm_per_prompt={"image": 15 * 5},
        )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=32,
    )

    new_content = ''
    image_cnt = 0
    for i, c in enumerate(content):
        if i == 0:
            assert c['type'] == 'text'
        if c['type'] == 'image_url':
            image_cnt += 1
            new_content += f'<image>\n'
        else:
            assert c['type'] == 'text'
            new_content += f'{c["text"]}\n'
    assert image_cnt == len(image_urls)

    prompt = f"<|User|>: {new_content}\n\n<|Assistant|>:"

    print('#### FINAL PROMPT ####')
    print(prompt)
    print('#######################')

    inputs = {
        "prompt": prompt,
        "multi_modal_data": {
            "image": [fetch_image(url) for url in image_urls]
        },
    }

    outputs = llm.generate(inputs, sampling_params=sampling_params)
    generated_answer = outputs[0].outputs[0].text
    return generated_answer


def run_internvl(content: List[dict], model: str) -> str:
    global llm

    image_urls = [c['image_url']['url'] for c in content if c['type'] == 'image_url']

    if not llm:
        llm = LLM(
            model=model,
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
            #max_model_len=4096,
            enforce_eager=True,
            tensor_parallel_size=torch.cuda.device_count(),
            quantization="AWQ" if "awq" in model.lower() else None,
            dtype="float16" if "awq" in model.lower() else "auto",
            max_num_seqs=1,
            limit_mm_per_prompt={"image": 15*5},
            mm_processor_kwargs={"max_dynamic_patch": 4},
        )

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=32,
        stop_token_ids=stop_token_ids,
    )

    new_content = ''
    image_cnt = 0
    for i, c in enumerate(content):
        if i == 0:
            assert c['type'] == 'text'
        if c['type'] == 'image_url':
            image_cnt += 1
            new_content += f'<image>\n'
        else:
            assert c['type'] == 'text'
            new_content += f'{c["text"]}\n'
    assert image_cnt == len(image_urls)

    messages = [{
        'role': 'user',
        'content': new_content
    }]

    tokenizer = AutoTokenizer.from_pretrained(model,
                                              trust_remote_code=True)
    prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)

    print('#### FINAL PROMPT ####')
    print(prompt)
    print('#######################')

    inputs = {
        "prompt": prompt,
        "multi_modal_data": {
            "image": [fetch_image(url) for url in image_urls]
        },
    }

    outputs = llm.generate(inputs, sampling_params=sampling_params)
    generated_answer = outputs[0].outputs[0].text
    return generated_answer


model_2_run = {
    "gpt-4.1-2025-04-14": run_openai,
    "gpt-4o-mini-2024-07-18": run_openai,
    'OpenGVLab/InternVL2-8B': run_internvl,
    "Qwen/Qwen2.5-VL-7B-Instruct": run_qwen2p5_vl,
    "Qwen/Qwen2.5-VL-32B-Instruct": run_qwen2p5_vl,
    "deepseek-ai/deepseek-vl2": run_deepseek_vl2,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--apikey', type=str, default=None)
    parser.add_argument('--train_file', type=str, default=None)
    parser.add_argument('--test_file', type=str, default=None)
    parser.add_argument('--videodir', type=str, default=None)
    parser.add_argument('--transcriptdir', type=str, default=None)
    parser.add_argument('--cachedir', type=str, default="Cache/")
    parser.add_argument('--min_transcript_segment_tokens', type=int, default=3)
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--max_frames', type=int, default=10)
    parser.add_argument('--outfile', type=str, default="Result/debug.jsonl")
    parser.add_argument('--model', type=str, default='gpt-4o-mini-2024-07-18')
    parser.add_argument("--no_visual", action="store_true")
    parser.add_argument("--no_transcript", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--no_fewshot_entity_restriction", action="store_true")
    parser.add_argument("--no_fewshot_embedding_search", action="store_true")
    parser.add_argument("--embedding_model", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--embed_transcript_factor", type=float, default=0.5)
    parser.add_argument("--embed_visual_factor", type=float, default=0.5)
    parser.add_argument("--resized_pixels", type=int, default=None)
    parser.add_argument("--debug_output", action="store_true")
    args = parser.parse_args()

    if args.model not in model_2_run:
        raise ValueError(f"Model type {args.model} is not supported. Supported models: {model_2_run.keys()}")

    if args.cachedir is not None:
        cache_dir = os.path.join(
            args.cachedir,
            f'mn-trnscrpt-sgmnt-tkns-{args.min_transcript_segment_tokens}_'
            f'mx-frms-{args.max_frames}_'
            f'rszd-pxcls-{args.resized_pixels}_'
            f'embddng-mdl-{args.embedding_model.split("/")[-1]}'
        )
        os.makedirs(cache_dir, exist_ok=True)
    else:
        cache_dir = None

    random.seed(args.seed)

    savedir = os.path.dirname(args.outfile)
    if savedir.strip() and not os.path.exists(savedir):
        os.makedirs(savedir)

    # Load train data for few-shot learning
    train_data_list: List[VideoData] = []
    if args.num_fewshot > 0:
        assert os.path.exists(args.train_file)
        train_data_list += load_dataset(
            fpath=args.train_file,
            video_dir=args.videodir,
            transcript_dir=args.transcriptdir,
            max_frames=args.max_frames,
            resized_pixels=args.resized_pixels,
            min_transcript_segment_tokens=args.min_transcript_segment_tokens,
            cache_dir=cache_dir,
        )
    # Load test data
    test_data_list = load_dataset(
        fpath=args.test_file,
        video_dir=args.videodir,
        transcript_dir=args.transcriptdir,
        max_frames=args.max_frames,
        resized_pixels=args.resized_pixels,
        min_transcript_segment_tokens=args.min_transcript_segment_tokens,
        cache_dir=cache_dir,
    )

    # Get list of the labels
    label_list = set([l for d in train_data_list + test_data_list for l in d.labels])
    label_list = sorted(label_list)

    # Load cached output data if exists
    existing_output_data, existing_output_data_ids = load_existing_output(args.outfile)

    # Load embedding model for the similarity search used in the few-shot learning
    embed_model = CLIPModel.from_pretrained(args.embedding_model)
    embed_processor = CLIPProcessor.from_pretrained(args.embedding_model)
    embed_tokenizer = AutoTokenizer.from_pretrained(args.embedding_model)

    if cache_dir is not None:
        print('Building and saving the dataset cache.')
        for data in tqdm.tqdm(train_data_list + test_data_list):
            cache_file = os.path.join(cache_dir, data.video_id + '.pkl')
            if os.path.exists(cache_file):
                continue
            data.load_frames_and_transcripts()
            data.load_embeddings(model=embed_model, processor=embed_processor, tokenizer=embed_tokenizer)
            data.save(cache_file)

    # Load the basic prompt text (where the annotation guideline is written)
    with open(args.prompt, 'r') as f:
        guideline_prompt = f.read()

    if args.num_fewshot > 0:
        # Create randomly sampled few-shot data from train data
        random_fewshot_data_list = random.sample(train_data_list, k=args.num_fewshot)

        # Create randomly sampled few-shot data from "in-domain" train data
        random_sampled_entity_train_data_list: Dict[str, List[VideoData]] = dict()
        domains = set([d.domain for d in train_data_list])  # We assume each entity as a domain
        for domain in domains:
            domain_train_data_list = [d for d in train_data_list if d.domain == domain]
            random_sampled_entity_train_data_list[domain] = random.sample(
                domain_train_data_list, k=min(args.num_fewshot, len(domain_train_data_list))
            )

    for data in test_data_list:
        if data.video_id in existing_output_data_ids:
            print(f'Skip inference for {data.video_id} because it exists in the output file.')
            continue

        print(f'[{data.video_id}] Start prediction')

        fewshot_data_list: List[VideoData] = []
        similarities = None
        if args.num_fewshot > 0:
            found_fewshot_data_list, similarities = get_fewshot_data(
                num_fewshot=args.num_fewshot,
                test_data=data,
                train_data_list=train_data_list,
                default_fewshot_data_list=random_fewshot_data_list,
                default_entity_fewshot_data_list=random_sampled_entity_train_data_list,
                model=embed_model, processor=embed_processor, tokenizer=embed_tokenizer,
                no_fewshot_entity_restriction=args.no_fewshot_entity_restriction,
                no_fewshot_embedding_search=args.no_fewshot_embedding_search,
                embed_transcript_factor=args.embed_transcript_factor,
                embed_visual_factor=args.embed_visual_factor,
            )
            fewshot_data_list = found_fewshot_data_list

        content = [
            {"type": "text", "text": guideline_prompt},
        ]

        for fewshot_data in fewshot_data_list:
            content.append({"type": "text", "text": 'Frame & transcript (if available) pairs of the video:'})
            for frame, segment in zip(fewshot_data.frames, fewshot_data.transcript_segments):
                if not args.no_visual:
                    content.append(
                        {'type': "image_url", "image_url": {"url": frame, "detail": "low"}}
                    )
                if not args.no_transcript:
                    content.append({"type": "text", "text": f"(Transcript) {segment['text']}\n"})
            jsonified_labels = json.dumps(fewshot_data.labels)
            content.append({"type": "text", "text": f"Answer: {jsonified_labels}"})

        content.append({"type": "text", "text": 'Frame & transcript (if available) pairs of the video:'})
        for frame, segment in zip(data.frames, data.transcript_segments):
            if not args.no_visual:
                content.append(
                    {'type': "image_url", "image_url": {"url": frame, "detail": "low"}}
                )
            if not args.no_transcript:
                content.append({"type": "text", "text": f"(Transcript) {segment['text']}\n"})
        content.append({"type": "text", "text": "Answer:"})

        print('#### GUIDELINE PROMPT ####')
        print(guideline_prompt)
        print('#######################')

        f_run = model_2_run[args.model]
        if f_run.__name__ == 'run_openai':
            generated_answer = f_run(content=content, apikey=args.apikey, model=args.model)
        else:
            generated_answer = f_run(content=content, model=args.model)

        pred_labels = parse_label(generated_answer=generated_answer, label_list=label_list)

        print('###### ANSWER ######')
        print(generated_answer)
        print('####################')

        print('###### PARSED ANSWER ######')
        print(pred_labels)
        print('###########################')

        existing_output_data.append({
            'video_id': data.video_id,
            'labels': pred_labels,
            'generated_answer': generated_answer,
            'seed': args.seed,
            'train_file': args.train_file,
            'test_file': args.test_file,
            'outfile': args.outfile,
            'videodir': args.videodir,
            'transcriptdir': args.transcriptdir,
            'min_transcript_segment_tokens': args.min_transcript_segment_tokens,
            'no_visual': args.no_visual,
            'no_transcript': args.no_transcript,
            'used_frames': len(data.frames),
            'model': args.model,
            'num_fewshot': args.num_fewshot,
            'fewshot_samples': [d.video_id for d in fewshot_data_list],
            'fewshot_sample_similarities': similarities if similarities else None,
            'resized_pixels': args.resized_pixels,
            'no_fewshot_entity_restriction': args.no_fewshot_entity_restriction,
            'no_fewshot_embedding_search': args.no_fewshot_embedding_search,
            'embed_transcript_factor': args.embed_transcript_factor,
            'embed_visual_factor': args.embed_visual_factor,
            'embedding_model': args.embedding_model,
        })

        with open(args.outfile, 'w') as f:
            for output_data in existing_output_data:
                f.write(json.dumps(output_data, ensure_ascii=False) + '\n')

        if args.debug_output:
            out_dir = os.path.join(
                os.path.dirname(args.outfile), f'{os.path.basename(args.outfile)}.debug_output', data.video_id
            )
            os.makedirs(out_dir, exist_ok=True)

            with open(os.path.join(out_dir, f'_prompt.html'), 'w') as f:
                f.write(f'<html>\n')
                f.write(f'<h2>Prompt for "{data.video_id}"</h2>\n')
                image_cnt = 0
                for c in content:
                    if c['type'] == 'image_url':
                        encoded = c['image_url']['url'].split('base64,')[1]
                        fname = f'image_{image_cnt}.jpg'
                        image_cnt += 1
                        with open(os.path.join(out_dir, fname), 'wb') as fi:
                            fi.write(base64.b64decode(encoded))
                        f.write(f'<img src="{fname}">\n')
                    else:
                        txt = c["text"].replace("\n", "<br>")
                        f.write(f'<p>{txt}</p>\n')
                f.write(f'<h2>Generated answer</h2>\n')
                f.write(f'<p>{generated_answer}</p>\n')
                f.write(f'</html>\n')


if __name__ == '__main__':
    main()

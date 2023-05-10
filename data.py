import copy
import json
import random
import shutil
from multiprocessing import cpu_count
from os.path import dirname
from typing import *

import numpy as np
import torch
from datasets import concatenate_datasets, Dataset, DownloadConfig, DownloadMode
from transformers import AutoProcessor, AutoTokenizer, PreTrainedTokenizer
from transformers.image_processing_utils import BaseImageProcessor

from coco_dataset import COCO


def make_data_block(
    samples: Dict[str, List[Any]],
    prompt_col_name: str,
    label_col_name: str,
    tokenizer: PreTrainedTokenizer,
    image_col_name: Optional[str] = None,
    processor: Optional[BaseImageProcessor] = None,
    preprocess_fn: Optional[Callable] = None,
    sample_max_len: int = 1024,
    block_max_len: int = 2048,
    add_eos_token: bool = False,
    truncate_prompt: bool = True,
    merge_prompt_label: bool = False
) -> Dict[str, List[torch.LongTensor]]:
    if preprocess_fn:
        samples = preprocess_fn(samples)

    prompts = samples[prompt_col_name]
    labels = [each if isinstance(each, str) else each[0] for each in samples[label_col_name]]
    if image_col_name:
        images = samples[image_col_name]
        assert processor, "processor can't be None if sample contains image"
        pixel_values = processor(images=images)["pixel_values"]
        for idx, pixel_value in enumerate(pixel_values):
            if isinstance(pixel_value, torch.Tensor):
                pixel_value = pixel_value.cpu().numpy()
            if isinstance(pixel_value, np.ndarray):
                pixel_value = pixel_value.tolist()
            pixel_values[idx] = pixel_value
    else:
        pixel_values = [None for _ in range(len(prompts))]

    # tokenize samples
    tokenized_prompts = tokenizer(prompts, truncation=False)["input_ids"]
    tokenized_labels = tokenizer(labels, truncation=False)["input_ids"]

    # filter tokenized samples by length
    dropped_indices = []
    for idx, (tokenized_prompt, tokenized_label) in enumerate(zip(tokenized_prompts, tokenized_labels)):
        if add_eos_token:
            tokenized_label += [tokenizer.eos_token_id]
        len_prompt = len(tokenized_prompt)
        len_label = len(tokenized_label)
        exceed_len = len_prompt + len_label - sample_max_len
        if exceed_len > 0:
            if truncate_prompt:
                tokenized_prompt = tokenized_prompt[exceed_len:]
            else:
                tokenized_label = tokenized_label[: -exceed_len]
        tokenized_prompts[idx] = tokenized_prompt
        tokenized_labels[idx] = tokenized_label
        if not tokenized_label:
            dropped_indices.append(idx)

    # make data blocks of samples
    tokenized_samples = sorted(
        [
            (p, l, pix) for idx, (p, l, pix) in enumerate(zip(tokenized_prompts, tokenized_labels, pixel_values))
            if idx not in dropped_indices
        ],
        key=lambda x: (len(x[0]) + len(x[1])) if merge_prompt_label else len(x[0])
    )
    sample_blocks = []
    sample_block = []
    blk_max_len = 0
    blk_total_len = 0
    for tokenized_sample in tokenized_samples:
        prompt_ids, label_ids, _ = tokenized_sample
        ori_sample_len = len(prompt_ids)
        if merge_prompt_label:
            ori_sample_len += len(label_ids)
        if ori_sample_len <= blk_max_len:
            additional_len = blk_max_len
            sample_len = blk_max_len
        else:
            additional_len = len(sample_block) * (ori_sample_len - blk_max_len) + ori_sample_len
            sample_len = ori_sample_len

        if blk_total_len + additional_len > block_max_len:
            sample_blocks.append((copy.copy(sample_block), blk_max_len))
            sample_block = []
            blk_max_len = 0
            blk_total_len = 0
            sample_len = ori_sample_len
            additional_len = ori_sample_len

        sample_block.append(tokenized_sample)
        blk_max_len = max(blk_max_len, sample_len)
        blk_total_len += additional_len

    if sample_block:
        sample_blocks.append((copy.copy(sample_block), blk_max_len))
    del sample_block
    del blk_max_len
    del blk_total_len

    new_samples = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
        "pixel_values": []
    }
    # padding each data block internally
    for block, blk_max_len in sample_blocks:
        input_ids = []
        attention_mask = []
        label_ids = []
        pixels = []
        label_max_len = max([len(sample[1]) for sample in block])

        for sample in block:
            tokenized_prompt, tokenized_label, pixel = sample
            sample_len = len(tokenized_prompt)
            if merge_prompt_label:
                sample_len += len(tokenized_label)
            pad_num = blk_max_len - sample_len
            if merge_prompt_label:
                input_ids.append([tokenizer.pad_token_id] * pad_num + tokenized_prompt + tokenized_label)
                label_ids.append([-100] * (pad_num + len(tokenized_prompt)) + tokenized_label)
            else:
                input_ids.append([tokenizer.pad_token_id] * pad_num + tokenized_prompt)
                label_ids.append([-100] * (label_max_len - len(tokenized_label)) + tokenized_label)
            attention_mask.append([0] * pad_num + [1] * sample_len)
            pixels.append(pixel)

        new_samples["input_ids"].append(input_ids)
        new_samples["attention_mask"].append(attention_mask)
        new_samples["labels"].append(label_ids)
        if pixels[0] is not None:
            new_samples["pixel_values"].append(pixels)
    if not new_samples["pixel_values"]:
        new_samples["pixel_values"] = [None for _ in range(len(new_samples["input_ids"]))]

    return new_samples


def collate_data(blocks: List[Dict[str, Optional[list]]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    def pad_block(block, pads):
        return torch.cat((pads.to(block.device), block), dim=-1)

    input_ids_blocks = [torch.LongTensor(block["input_ids"]) for block in blocks]
    attention_mask_blocks = [torch.LongTensor(block["attention_mask"]) for block in blocks]
    label_blocks = [torch.LongTensor(block["labels"]) for block in blocks]
    if blocks[0]["pixel_values"] is not None:
        pixel_blocks = torch.cat([torch.FloatTensor(block["pixel_values"]) for block in blocks], dim=0)
    else:
        pixel_blocks = None

    bsz = len(blocks)
    inp_max_len = max([block.size(-1) for block in input_ids_blocks])
    label_max_len = max([block.size(-1) for block in label_blocks])

    for i in range(bsz):
        block_bsz, block_inp_len = input_ids_blocks[i].shape
        block_label_len = label_blocks[i].shape[-1]
        pad_num = inp_max_len - block_inp_len
        if pad_num > 0:
            input_ids_blocks[i] = pad_block(input_ids_blocks[i], torch.ones((block_bsz, pad_num)) * pad_token_id)
            attention_mask_blocks[i] = pad_block(attention_mask_blocks[i], torch.zeros((block_bsz, pad_num)))
        label_pad_num = label_max_len - block_label_len
        if label_pad_num > 0:
            label_blocks[i] = pad_block(label_blocks[i], torch.ones((block_bsz, label_pad_num)) * -100)

    outputs = {
        "input_ids": torch.cat(input_ids_blocks, dim=0).long(),
        "attention_mask": torch.cat(attention_mask_blocks, dim=0).long(),
        "labels": torch.cat(label_blocks, dim=0).long(),
        "pixel_values": pixel_blocks
    }

    return outputs


def load_coco_dataset():
    ds = COCO(name="2014_captions")
    ds.download_and_prepare(download_config=DownloadConfig(delete_extracted=True), download_mode=DownloadMode.FORCE_REDOWNLOAD)
    dl_manager = ds.dl_manager
    ds = ds.as_dataset(in_memory=True)

    return ds, dl_manager


def load_gpt4llm():
    with open("datasets/gpt4llm/alpaca_gpt4_data.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    def gen_fn():
        return raw_data

    ds = Dataset.from_generator(gen_fn, keep_in_memory=True)
    ds.cleanup_cache_files()
    return ds


def load_dolly():
    with open("datasets/dolly/databriks_dolly_15k.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    def gen_fn():
        return raw_data

    ds = Dataset.from_generator(gen_fn, keep_in_memory=True)
    ds.cleanup_cache_files()
    return ds


def load_baize_chat():
    with open("datasets/baize_chat/baize_chat_quora.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    def gen_fn():
        return raw_data

    ds = Dataset.from_generator(gen_fn, keep_in_memory=True)
    ds.cleanup_cache_files()
    return ds


def build_image_caption_dataset(
    pretrained_vision_model_name_or_path=None,
    pretrained_language_model_name_or_path=None,
    processor=None,
    tokenizer=None,
    use_fast_tokenizer: bool = False,
    instruction_prefix: str = "<|SYSTEM|> ",
    output_prefix: str = "<|ASSISTANT|> ",
    spliter: str = "\n\n",
    instruction: str = "describe the image",
    sample_max_len: int = 256,
    block_max_len: int = 256,
    add_eos_token: bool = False,
    seed: int = 1024,
    num_train_samples: int = 50000,
    num_eval_samples: int = 5000,
    for_inference: bool = False,
):
    random.seed(seed)

    if not processor:
        processor = AutoProcessor.from_pretrained(pretrained_vision_model_name_or_path)
    if not tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_language_model_name_or_path, use_fast=use_fast_tokenizer)
        if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    ds, dl_manager = load_coco_dataset()
    train_ds, validation_ds = ds["train"], ds["validation"]
    train_ds = train_ds.select(
        indices=random.sample(list(range(len(train_ds))), k=min(len(train_ds), num_train_samples)),
        keep_in_memory=True
    )
    validation_ds = validation_ds.select(
        indices=random.sample(list(range(len(validation_ds))), k=min(len(validation_ds), num_eval_samples)),
        keep_in_memory=True
    )
    prompt = instruction_prefix + instruction + spliter + output_prefix
    train_ds = train_ds.add_column("prompt", [prompt for _ in range(len(train_ds))])
    validation_ds = validation_ds.add_column("prompt", [prompt for _ in range(len(validation_ds))])

    n_proc = cpu_count() - 2
    train_ds = train_ds.map(
        make_data_block,
        batched=True,
        batch_size=min(100, len(train_ds) // n_proc),
        num_proc=n_proc,
        remove_columns=train_ds.column_names,
        keep_in_memory=True,
        load_from_cache_file=False,
        fn_kwargs={
            "prompt_col_name": "prompt",
            "label_col_name": "sentences_raw",
            "image_col_name": "image",
            "tokenizer": tokenizer,
            "processor": processor,
            "sample_max_len": sample_max_len,
            "block_max_len": block_max_len,
            "add_eos_token": add_eos_token,
            "truncate_prompt": False,
            "merge_prompt_label": not for_inference
        }
    )
    validation_ds = validation_ds.map(
        make_data_block,
        batched=True,
        batch_size=min(100, len(train_ds) // n_proc),
        num_proc=n_proc,
        remove_columns=validation_ds.column_names,
        keep_in_memory=True,
        load_from_cache_file=False,
        fn_kwargs={
            "prompt_col_name": "prompt",
            "label_col_name": "sentences_raw",
            "image_col_name": "image",
            "tokenizer": tokenizer,
            "processor": processor,
            "sample_max_len": sample_max_len,
            "block_max_len": block_max_len,
            "add_eos_token": add_eos_token,
            "truncate_prompt": False,
            "merge_prompt_label": not for_inference
        }
    )

    shutil.rmtree(dirname(list(dl_manager.extracted_paths.values())[0]))

    return train_ds, validation_ds


def build_instruction_following_dataset(
    pretrained_model_name_or_path=None,
    tokenizer=None,
    instruction_prefix: str = "<|SYSTEM|> ",
    input_prefix: str = "<|USER|> ",
    output_prefix: str = "<|ASSISTANT|> ",
    task_type_prefix: Optional[str] = None,
    spliter: str = "\n\n",
    sample_max_len: int = 1024,
    block_max_len: int = 1024,
    add_eos_token: bool = False,
    use_fast_tokenizer: bool = False,
    seed: int = 1024,
    num_train_blocks: int = 10000,
    num_eval_blocks: int = 1000,
    for_inference: bool = False
):
    random.seed(seed)

    def preprocess_fn(examples):
        instruction_data = examples["instruction"]
        input_data = examples["input"]
        output_data = examples["output"]
        task_type_data = examples.get("task_type", [None for _ in range(len(instruction_data))])

        new_examples = {
            "input": [],
            "output": []
        }
        for instruction, input_, output, task_type in zip(instruction_data, input_data, output_data, task_type_data):
            new_input = instruction_prefix + instruction + spliter
            if input_:
                new_input += input_prefix + input_ + spliter
            new_input += output_prefix
            if task_type and task_type_prefix:
                new_input = task_type_prefix + task_type + spliter + new_input
            new_examples["input"].append(new_input)
            new_examples["output"].append(output)

        return new_examples

    if not tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, use_fast=use_fast_tokenizer)
        if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    ds_gpt4llm = load_gpt4llm()
    ds_dolly = load_dolly()
    n_proc = cpu_count() - 2
    ds_gpt4llm = ds_gpt4llm.map(
        make_data_block,
        batched=True,
        batch_size=min(1000, len(ds_gpt4llm) // n_proc),
        num_proc=n_proc,
        remove_columns=ds_gpt4llm.column_names,
        keep_in_memory=True,
        load_from_cache_file=False,
        fn_kwargs={
            "prompt_col_name": "input",
            "label_col_name": "output",
            "tokenizer": tokenizer,
            "preprocess_fn": preprocess_fn,
            "sample_max_len": sample_max_len,
            "block_max_len": block_max_len,
            "add_eos_token": add_eos_token,
            "truncate_prompt": False,
            "merge_prompt_label": not for_inference
        }
    )
    ds_dolly = ds_dolly.map(
        make_data_block,
        batched=True,
        batch_size=min(1000, len(ds_dolly) // n_proc),
        num_proc=n_proc,
        remove_columns=ds_dolly.column_names,
        keep_in_memory=True,
        load_from_cache_file=False,
        fn_kwargs={
            "prompt_col_name": "input",
            "label_col_name": "output",
            "tokenizer": tokenizer,
            "preprocess_fn": preprocess_fn,
            "sample_max_len": sample_max_len,
            "block_max_len": block_max_len,
            "add_eos_token": add_eos_token,
            "truncate_prompt": False,
            "merge_prompt_label": not for_inference
        }
    )
    ds = concatenate_datasets([ds_gpt4llm, ds_dolly])
    split_ds = ds.train_test_split(test_size=len(ds) // 10, seed=seed)
    train_ds, validation_ds = split_ds["train"], split_ds["test"]
    train_ds = train_ds.select(
        indices=random.sample(list(range(len(train_ds))), k=min(len(train_ds), num_train_blocks)),
        keep_in_memory=True
    )
    validation_ds = validation_ds.select(
        indices=random.sample(list(range(len(validation_ds))), k=min(len(validation_ds), num_eval_blocks)),
        keep_in_memory=True
    )

    return train_ds, validation_ds


def build_baize_chat_dataset(
    pretrained_model_name_or_path=None,
    tokenizer=None,
    spliter: str = "\n\n",
    max_utterances: int = 20,
    instruction_prefix: str = "<|SYSTEM|> ",
    input_prefix: str = "<|USER|> ",
    output_prefix: str = "<|ASSISTANT|> ",
    sample_max_len: int = 1024,
    block_max_len: int = 1024,
    add_eos_token: bool = False,
    use_fast_tokenizer: bool = False,
    seed: int = 1024,
    num_train_blocks: int = 10000,
    num_eval_blocks: int = 1000,
    for_inference: bool = False
):
    human_symbol = "[|Human|]"
    ai_symbol = "[|AI|]"
    bkg_template = (
        "I want you to be an AI Assistant. You are very helpful and polite. "
        "You are now talking with a Human about {topic}"
    )

    def preprocess_fn(examples):
        topic_col_data = examples["topic"]
        context_col_data = examples["input"]

        new_examples = {
            "input": [],
            "output": []
        }
        for topic, raw_context in zip(topic_col_data, context_col_data):
            lines = raw_context.split("\n")
            bkg = bkg_template.format(topic=topic)

            if not lines[0].startswith(human_symbol) and not lines[0].startswith(ai_symbol):
                # drop dataset's original conversation background
                lines = lines[1:]
            if not lines[-1].replace(human_symbol, "").strip() or not lines[-1].replace(ai_symbol, "").strip():
                lines = lines[:-1]
            if not lines[-1].startswith(ai_symbol):
                lines = lines[:-1]

            utterances = []
            for line in lines[: -1]:
                if line.startswith(human_symbol):
                    utterances.append(input_prefix + line[len(human_symbol):].strip())
                elif line.startswith(ai_symbol):
                    utterances.append(output_prefix + line[len(ai_symbol):].strip())
                else:
                    utterances[-1] = utterances[-1] + f" {line.strip()}"

            bkg = instruction_prefix + bkg + spliter
            if random.uniform(0, 1) <= 0.5:
                bkg = ""
            new_examples["input"].append(
                bkg + spliter.join(utterances[: max_utterances]) + spliter + output_prefix
            )
            new_examples["output"].append(" " + lines[-1][len(ai_symbol):].strip())

        return new_examples

    if not tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, use_fast=use_fast_tokenizer)
        if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    ds = load_baize_chat()
    n_proc = cpu_count() - 2
    ds = ds.map(
        make_data_block,
        batched=True,
        batch_size=min(1000, len(ds) // n_proc),
        num_proc=n_proc,
        remove_columns=ds.column_names,
        keep_in_memory=True,
        load_from_cache_file=False,
        fn_kwargs={
            "prompt_col_name": "input",
            "label_col_name": "output",
            "tokenizer": tokenizer,
            "preprocess_fn": preprocess_fn,
            "sample_max_len": sample_max_len,
            "block_max_len": block_max_len,
            "add_eos_token": add_eos_token,
            "truncate_prompt": True,
            "merge_prompt_label": not for_inference
        }
    )

    split_ds = ds.train_test_split(test_size=len(ds) // 10, seed=seed)
    train_ds, validation_ds = split_ds["train"], split_ds["test"]
    train_ds = train_ds.select(
        indices=random.sample(list(range(len(train_ds))), k=min(len(train_ds), num_train_blocks)),
        keep_in_memory=True
    )
    validation_ds = validation_ds.select(
        indices=random.sample(list(range(len(validation_ds))), k=min(len(validation_ds), num_eval_blocks)),
        keep_in_memory=True
    )

    return train_ds, validation_ds


def build_dataset(
    pretrained_language_model_name_or_path,
    pretrained_vision_model_name_or_path,
    processor=None,
    tokenizer=None,
    use_fast_tokenizer: bool = False,
    task_type_prefix: str = "<|TASK_TYPE|> ",
    instruction_prefix: str = "<|SYSTEM|> ",
    input_prefix: str = "<|USER|> ",
    output_prefix: str = "<|ASSISTANT|> ",
    spliter: str = "\n\n",
    caption_instruction: str = "describe the image",
    chat_utterance_max_num: int = 20,
    image_caption_sample_max_len: int = 256,
    image_caption_block_max_len: int = 256,
    instruction_following_sample_max_len: int = 1024,
    instruction_following_block_max_len: int = 1024,
    chat_sample_max_len: int = 1024,
    chat_block_max_len: int = 1024,
    add_eos_token: bool = False,
    seed: int = 1024,
    num_image_caption_train_samples: int = 50000,
    num_image_caption_eval_samples: int = 5000,
    num_instruction_following_train_blocks: int = 10000,
    num_instruction_following_eval_blocks: int = 1000,
    num_chat_train_blocks: int = 10000,
    num_chat_eval_blocks: int = 1000,
    for_inference: bool = False
):
    instruction_following_train_ds, instruction_following_validation_ds = build_instruction_following_dataset(
        pretrained_model_name_or_path=pretrained_language_model_name_or_path,
        tokenizer=tokenizer,
        task_type_prefix=task_type_prefix,
        instruction_prefix=instruction_prefix,
        input_prefix=input_prefix,
        output_prefix=output_prefix,
        spliter=spliter,
        sample_max_len=instruction_following_sample_max_len,
        block_max_len=instruction_following_block_max_len,
        add_eos_token=add_eos_token,
        use_fast_tokenizer=use_fast_tokenizer,
        seed=seed,
        num_train_blocks=num_instruction_following_train_blocks,
        num_eval_blocks=num_instruction_following_eval_blocks,
        for_inference=for_inference
    )
    baize_chat_train_ds, baize_chat_validation_ds = build_baize_chat_dataset(
        pretrained_model_name_or_path=pretrained_language_model_name_or_path,
        tokenizer=tokenizer,
        spliter=spliter,
        max_utterances=chat_utterance_max_num,
        instruction_prefix=instruction_prefix,
        input_prefix=input_prefix,
        output_prefix=output_prefix,
        sample_max_len=chat_sample_max_len,
        block_max_len=chat_block_max_len,
        add_eos_token=add_eos_token,
        use_fast_tokenizer=use_fast_tokenizer,
        seed=seed,
        num_train_blocks=num_chat_train_blocks,
        num_eval_blocks=num_chat_eval_blocks,
        for_inference=for_inference
    )
    image_caption_train_ds, image_caption_validation_ds = build_image_caption_dataset(
        pretrained_language_model_name_or_path=pretrained_language_model_name_or_path,
        pretrained_vision_model_name_or_path=pretrained_vision_model_name_or_path,
        processor=processor,
        tokenizer=tokenizer,
        use_fast_tokenizer=use_fast_tokenizer,
        instruction_prefix=instruction_prefix,
        output_prefix=output_prefix,
        spliter=spliter,
        instruction=caption_instruction,
        sample_max_len=image_caption_sample_max_len,
        block_max_len=image_caption_block_max_len,
        add_eos_token=add_eos_token,
        seed=seed,
        num_train_samples=num_image_caption_train_samples,
        num_eval_samples=num_image_caption_eval_samples,
        for_inference=for_inference
    )

    train_ds = concatenate_datasets(
        [image_caption_train_ds, instruction_following_train_ds, baize_chat_train_ds]
    ).shuffle(seed=seed)
    validation_ds = concatenate_datasets(
        [image_caption_validation_ds, instruction_following_validation_ds, baize_chat_validation_ds]
    )

    return train_ds, validation_ds

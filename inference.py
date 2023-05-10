from argparse import ArgumentParser
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoTokenizer, PreTrainedTokenizer

from clip import ClipAdaptionPromptV2ForMultiModalConditionalGeneration
from data import build_dataset, collate_data


@torch.no_grad()
@torch.cuda.amp.autocast()
def inference():
    parser = ArgumentParser()
    parser.add_argument("--pretrained_language_model_name_or_path", type=str)
    parser.add_argument("--pretrained_vision_model_name_or_path", type=str)
    parser.add_argument("--adapters_save_dir", type=str)
    args = parser.parse_args()

    model = ClipAdaptionPromptV2ForMultiModalConditionalGeneration.from_pretrained(
        pretrained_language_model_name_or_path=args.pretrained_language_model_name_or_path,
        pretrained_vision_model_name_or_path=args.pretrained_vision_model_name_or_path,
        adapters_save_dir=args.adapters_save_dir,
        language_model_loading_kwargs={"load_in_8bit": False, "low_cpu_mem_usage": True, "device_map": "auto", "torch_dtype": torch.float16},
        vision_model_loading_kwargs={"load_in_8bit": False, "low_cpu_mem_usage": True, "torch_dtype": torch.float16},
    )
    model.eval()

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(args.pretrained_language_model_name_or_path, use_fast=False)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    processor = AutoProcessor.from_pretrained(args.pretrained_vision_model_name_or_path)
    _, eval_ds = build_dataset(
        pretrained_language_model_name_or_path=args.pretrained_language_model_name_or_path,
        pretrained_vision_model_name_or_path=args.pretrained_vision_model_name_or_path,
        tokenizer=tokenizer,
        processor=processor,
        image_caption_sample_max_len=128,
        image_caption_block_max_len=192,
        instruction_following_sample_max_len=1024,
        instruction_following_block_max_len=1024,
        num_image_caption_train_samples=1,
        num_image_caption_eval_samples=5,
        num_instruction_following_train_blocks=1,
        num_instruction_following_eval_blocks=5,
        for_inference=True
    )

    dl = DataLoader(
        eval_ds, batch_size=1, shuffle=False, collate_fn=partial(collate_data, pad_token_id=tokenizer.pad_token_id)
    )

    for batch in dl:
        labels = batch.pop("labels")
        labels = torch.where(labels == -100, torch.ones_like(labels) * tokenizer.pad_token_id, labels)
        labels = tokenizer.batch_decode(labels, clean_up_tokenization_spaces=True)
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                if k == "pixel_values":
                    v = v.to(torch.float16)
                batch[k] = v.to(model.vision_model.device)
        outputs = model.generate(**batch, **{"num_beams": 1, "max_new_tokens": 64, "do_sample": False})
        outputs = tokenizer.batch_decode(outputs, clean_up_tokenization_spaces=True)
        for label, output in zip(labels, outputs):
            print(f"label: {label.lstrip('</s>').strip()}")
            print(f"model: {output.lstrip('</s>').strip()}")
            print("=" * 42)


if __name__ == "__main__":
    inference()

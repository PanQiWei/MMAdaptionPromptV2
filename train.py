import math
from argparse import ArgumentParser
from functools import partial
from os.path import join

import torch
from clip import ClipAdaptionPromptV2ForMultiModalConditionalGeneration
from data import build_dataset, collate_data
from peft import AdaptionPromptV2Config, TaskType, PeftType
from transformers import AutoProcessor, AutoTokenizer, Trainer, TrainingArguments


def train():
    parser = ArgumentParser()
    parser.add_argument("--pretrained_language_model_name_or_path", type=str)
    parser.add_argument("--pretrained_vision_model_name_or_path", type=str)
    args = parser.parse_args()

    pretrained_language_model_name_or_path = args.pretrained_language_model_name_or_path
    pretrained_vision_model_name_or_path = args.pretrained_vision_model_name_or_path
    train_args = TrainingArguments(
        output_dir=join(pretrained_language_model_name_or_path, "mm_adaption_prompt_v2"),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        learning_rate=6e-3,
        num_train_epochs=3,
        deepspeed=None,
        gradient_checkpointing=False,
        gradient_accumulation_steps=4,
        logging_strategy="steps",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="no",
        save_total_limit=1,
        load_best_model_at_end=False,
        disable_tqdm=False,
        remove_unused_columns=False,
        local_rank=-1,
        do_train=True,
        do_eval=True,
        seed=1024,
        data_seed=1024,
        fp16=True,
        fp16_full_eval=True,
        bf16=False,
        bf16_full_eval=False
    )
    train_args.gradient_accumulation_steps = max(1, train_args.gradient_accumulation_steps // train_args.world_size)
    print("preparing tokenizer and processor...")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_language_model_name_or_path, use_fast=False)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    processor = AutoProcessor.from_pretrained(pretrained_vision_model_name_or_path)
    print("preparing datasets...")
    train_ds, eval_ds = build_dataset(
        pretrained_language_model_name_or_path=pretrained_language_model_name_or_path,
        pretrained_vision_model_name_or_path=pretrained_vision_model_name_or_path,
        tokenizer=tokenizer,
        processor=processor,
        chat_utterance_max_num=20,
        image_caption_sample_max_len=128,
        image_caption_block_max_len=192,
        instruction_following_sample_max_len=1024,
        instruction_following_block_max_len=1024,
        chat_sample_max_len=1024,
        chat_block_max_len=1024,
        num_image_caption_train_samples=100000,
        num_image_caption_eval_samples=5000,
        num_instruction_following_train_blocks=50000,
        num_instruction_following_eval_blocks=5000,
        num_chat_train_blocks=50000,
        num_chat_eval_blocks=5000,
    )
    print("preparing model...")
    model = ClipAdaptionPromptV2ForMultiModalConditionalGeneration.build_model_for_train(
        pretrained_language_model_name_or_path=pretrained_language_model_name_or_path,
        pretrained_vision_model_name_or_path=pretrained_vision_model_name_or_path,
        hf_train_args=train_args,
        adaption_prompt_v2_config=AdaptionPromptV2Config(
            peft_type=PeftType.ADAPTION_PROMPT_V2,
            task_type=TaskType.CAUSAL_LM,
            adapter_len=10,
            adapter_layers=30,
            add_bias=True,
            add_scale=True,
            multi_modal=True
        ),
        language_model_loading_kwargs={"load_in_8bit": False, "low_cpu_mem_usage": True, "device_map": "auto", "torch_dtype": torch.float16},
        vision_model_loading_kwargs={"load_in_8bit": False, "low_cpu_mem_usage": True, "torch_dtype": torch.float16},
    )
    print("preparing trainer...")
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=partial(collate_data, pad_token_id=tokenizer.pad_token_id)
    )

    print(f"{train_args.parallel_mode=}")
    print(f"{train_args.per_device_train_batch_size=}")
    print(f"{train_args.gradient_accumulation_steps=}")
    print(f"epoch_steps={len(trainer.get_train_dataloader())}")
    print("training...")

    train_res = trainer.train()
    train_metrics = train_res.metrics
    train_ppl = math.exp(train_metrics["train_loss"])

    print("evaluating...")

    eval_metrics = trainer.evaluate()
    eval_ppl = math.exp(eval_metrics["eval_loss"])

    print(f"{train_ppl=}\t{eval_ppl=}")

    model.save_pretrained(train_args.output_dir)


if __name__ == "__main__":
    train()

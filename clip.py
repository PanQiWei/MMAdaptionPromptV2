from os import makedirs
from os.path import join
from typing import *

import torch
import torch.nn as nn
from peft import get_peft_model, prepare_model_for_int8_training, PeftModel
from peft.tuners.adaption_prompt_v2 import AdaptionPromptV2Model, AdaptionPromptV2Config
from transformers import AutoModelForCausalLM, CLIPVisionModel, TrainingArguments


class ClipAdaptionPromptV2ForMultiModalConditionalGeneration(nn.Module):
    def __init__(
        self,
        language_model: AdaptionPromptV2Model,
        vision_model: CLIPVisionModel
    ):
        super(ClipAdaptionPromptV2ForMultiModalConditionalGeneration, self).__init__()
        assert language_model._enabled
        assert language_model._configs[language_model._active_adapter].multi_modal
        assert "vision" in language_model._configs[language_model._active_adapter].supported_modals
        self.language_model = language_model
        self.vision_model = vision_model
        self.visual_projection = nn.Linear(
            vision_model.config.hidden_size,
            language_model.model.config.hidden_size,
            bias=False
        )
        self.visual_projection.to(self.vision_model.device)
        self._freeze()

    def _freeze(self):
        for n, p in self.vision_model.named_parameters():
            p.requires_grad = False

    def get_language_model_input_embeddings(self):
        if hasattr(self.language_model.model, "get_input_embeddings"):
            text_embeddings_module = getattr(self.language_model.model, "get_input_embeddings")()
        else:
            text_embeddings_module = getattr(
                getattr(self.language_model.model, self.language_model.model.config.model_type),
                "get_input_embeddings"
            )()
        return text_embeddings_module

    def get_text_input_features(self, input_ids: torch.LongTensor):
        return self.get_language_model_input_embeddings()(input_ids)

    def get_image_input_features(self, pixel_values: torch.FloatTensor):
        return self.visual_projection(self.vision_model(pixel_values)[1].unsqueeze(1))

    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        image_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        train_mode: bool = False
    ):
        if inputs_embeds is None:
            inputs_embeds = self.get_text_input_features(input_ids)
        if pixel_values is not None:
            if image_embeds is None:
                image_embeds = self.get_image_input_features(pixel_values)
            else:
                if len(image_embeds.shape) == 2:
                    image_embeds = image_embeds.unsqueeze(1)
            image_embeds = image_embeds.type_as(inputs_embeds)
            image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
            inputs_embeds = torch.cat([image_embeds, inputs_embeds], dim=1)
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids).to(image_embeds.device)
            attention_mask = torch.cat([image_attention_mask, attention_mask], dim=1)
            if labels is not None:
                labels = torch.cat([image_attention_mask * -100, labels], dim=1)

        if train_mode:
            self.language_model.freeze_adaption_params(None if pixel_values is not None else "vision")
            self.language_model.unfreeze_adaption_params(None if pixel_values is None else "vision")

        outputs = self.language_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=None,
            inputs_embeds=inputs_embeds,
            labels=labels
        )

        return outputs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        pixel_values=None,
        past_key_values=None,
        attention_mask=None
    ):
        inputs = self.language_model.prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask
        )
        text_embeds = self.get_text_input_features(inputs["input_ids"])
        inputs["inputs_embeds"] = text_embeds
        if pixel_values is not None:
            image_embeds = self.get_image_input_features(pixel_values)
            image_embeds = image_embeds.type_as(text_embeds)
            image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
            inputs["inputs_embeds"] = torch.cat([image_embeds, text_embeds], dim=1)
            inputs["attention_mask"] = torch.cat([image_attention_mask, inputs["attention_mask"]], dim=1)
        inputs["input_ds"] = None
        if "position_ids" in inputs:
            del inputs["position_ids"]

        return inputs

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        pixel_values=None,
        past_key_values=None,
        attention_mask=None,
        **kwargs
    ):
        inputs = self.prepare_inputs_for_generation(
            input_ids=input_ids,
            pixel_values=pixel_values,
            past_key_values=past_key_values,
            attention_mask=attention_mask
        )
        inputs = {k: v for k, v in inputs.items() if v is not None}
        return self.language_model.generate(**inputs, **kwargs)

    def save_pretrained(self, save_dir: str):
        makedirs(save_dir, exist_ok=True)

        self.language_model.save_pretrained(save_dir)
        torch.save(self.visual_projection.state_dict(), join(save_dir, "visual_projection.bin"))

    @classmethod
    def from_pretrained(
        cls,
        pretrained_language_model_name_or_path: str,
        pretrained_vision_model_name_or_path: str,
        adapters_save_dir: str,
        adapter_name: str = None,
        language_model_loading_kwargs: Optional[dict] = None,
        vision_model_loading_kwargs: Optional[dict] = None,
    ):
        language_model = AutoModelForCausalLM.from_pretrained(
            pretrained_language_model_name_or_path,
            **language_model_loading_kwargs
        )
        language_model = PeftModel.from_pretrained(
            model=language_model,
            model_id=adapters_save_dir,
            adapter_name=adapter_name or "default"
        )
        vision_model = CLIPVisionModel.from_pretrained(
            pretrained_vision_model_name_or_path,
            **vision_model_loading_kwargs
        )
        vision_model.to(language_model.device)

        model = cls(language_model, vision_model)
        model.visual_projection.load_state_dict(
            torch.load(
                join(adapters_save_dir, "visual_projection.bin"),
                map_location=next(iter(model.visual_projection.parameters())).device
            ),
        )

        return model

    @classmethod
    def build_model_for_train(
        cls,
        pretrained_language_model_name_or_path: str,
        pretrained_vision_model_name_or_path: str,
        adaption_prompt_v2_config: AdaptionPromptV2Config,
        hf_train_args: TrainingArguments,
        language_model_loading_kwargs: Optional[dict] = None,
        vision_model_loading_kwargs: Optional[dict] = None,
    ):
        if not language_model_loading_kwargs:
            language_model_loading_kwargs = dict()
        if not vision_model_loading_kwargs:
            vision_model_loading_kwargs = dict()

        language_model = AutoModelForCausalLM.from_pretrained(
            pretrained_language_model_name_or_path,
            **language_model_loading_kwargs
        )
        if language_model_loading_kwargs.get("load_in_8bit", True):
            language_model = prepare_model_for_int8_training(language_model, hf_train_args.gradient_checkpointing)
        language_model = get_peft_model(
            model=language_model,
            peft_config=adaption_prompt_v2_config
        )
        vision_model = CLIPVisionModel.from_pretrained(
            pretrained_vision_model_name_or_path,
            **vision_model_loading_kwargs
        )
        vision_model.to(language_model.device)
        if vision_model_loading_kwargs.get("load_in_8bit", True):
            vision_model = prepare_model_for_int8_training(vision_model, hf_train_args.gradient_checkpointing)
        model = cls(
            language_model=language_model,
            vision_model=vision_model
        )
        nn.init.normal_(
            model.visual_projection.weight,
            std=model.vision_model.config.hidden_size ** -0.5 * model.vision_model.config.initializer_factor,
        )
        model.train()

        return model

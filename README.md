# MMAdaptionPromptV2
an attempt to implement multi-modal llama-adapter-v2 that compatible with more other models.

## Notes
- use [this peft fork](https://github.com/PanQiWei/peft/tree/multi_modal_adaption_prompt) which implements prompt-adaption-v2 that supports multi-modal fine-tuning.
- currently only trained using clip-large + llama-7b on 1xA100-40G.

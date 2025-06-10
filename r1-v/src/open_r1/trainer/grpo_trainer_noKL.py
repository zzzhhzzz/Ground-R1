# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from PIL import Image,ImageDraw
from datetime import datetime
import re

import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union

import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url

from trainer.image_pro import process_vision_info

import copy
import gc
import numpy as np



if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list, list], list[float]]]
ScoreFunc = Union[str, PreTrainedModel, Callable[[list, list, list, list], list[float]]]

STAGE_ONE_TEMPLATE = (
    "Question: {Question}\n"
    "Please think about this question as if you were a human pondering deeply. "
    "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
    "It's encouraged to include self-reflection or verification in the reasoning process."
    "Give one bounding box coordinate of the region that can help you answer the question better. Following [x1,y1,x2,y2] format."
    "The size of the image: Width:{input_width}, Height:{input_height}."
    "Provide detailed reasoning between the <think> </think> tags first, then give the bounding box between the <box> </box> tags, finally give your final answer between the <answer> </answer> tags."
    "Format Example: <think> Reasoning process </think><box>[x1,y1,x2,y2]</box><answer> Final answer </answer>"
)

STAGE_TWO_TEMPLATE = (
    "Question: {Question}\n"
    "You can see the original image and the cropped image based on the bounding box you provided earlier."
    "You might find the reasoning process, bounding box, and answer you previously provided are not entirely correct or complete."
    "By referring to the original image and the image cropped from your previous bounding box, please try to provide a more accurate bounding box applied to the original image that can help you answer the question better. Following [x1,y1,x2,y2] format. Also, provide your updated reasoning and answer."
    "The size of the original image: Width:{input_width}, Height:{input_height}."
    "Provide detailed reasoning between the <think> </think> tags first, then give the bounding box between the <box> </box> tags, finally give your final answer between the <answer> </answer> tags."
    "Format Example: <think> Reasoning process </think><box>[x1,y1,x2,y2]</box><answer> Final answer </answer>"
)

STAGE_THREE_TEMPLATE = (
    "Question: {Question}\n"
    "You can see the original image and the cropped images based on the bounding boxes you provided earlier."
    "You might find the reasoning processes, bounding boxes, and answers you previously provided are not entirely correct or complete."
    "By referring to the original image and the images cropped from your previous bounding boxes, please try to provide a more accurate bounding box applied to the original image that can help you answer the question better. Following [x1,y1,x2,y2] format. Also, provide your updated reasoning and answer."
    "The size of the original image: Width:{input_width}, Height:{input_height}."
    "Provide detailed reasoning between the <think> </think> tags first, then give the bounding box between the <box> </box> tags, finally give your final answer between the <answer> </answer> tags."
    "Format Example: <think> Reasoning process </think><box>[x1,y1,x2,y2]</box><answer> Final answer </answer>"
)

def bbox_adjust(bbox, input_width, input_height, width, height, min_size=28):
    x1, y1, x2, y2 = bbox

    if x1 > input_width or x2 > input_width or y1 > input_height or y2 > input_height:
        print(f"Bounding Box out of the input image---{bbox}--input_w{input_width}--w{width}--input_h{input_height}--h{height}")

    x1 = min(x1, input_width)
    x2 = min(x2, input_width)
    y1 = min(y1, input_height)
    y2 = min(y2, input_height)

    
    y1 = int(y1/input_height * height)
    x1 = int(x1/input_width * width)
    y2 = int(y2/input_height * height)
    x2 = int(x2/input_width * width)

    bbox_width = x2 - x1
    bbox_height = y2 - y1
    
    if bbox_width < min_size:
        s = (min_size - bbox_width) // 2
        x1 -= s
        x2 += s
        if x2 - x1 < min_size:
            x2 += 1
    
    if bbox_height < min_size:
        s = (min_size - bbox_height) // 2
        y1 -= s
        y2 += s
        if y2 - y1 < min_size:
            y2 += 1

    return [x1, y1, x2, y2]

def cal_bbox_for_iou(bbox, input_width, input_height, width, height):
    if bbox:
        x1, y1, x2, y2 = bbox
        y1 = int(y1/input_height * height)
        x1 = int(x1/input_width * width)
        y2 = int(y2/input_height * height)
        x2 = int(x2/input_width * width)
        return [x1, y1, x2, y2]
    else:
        return bbox

class Qwen2VLGRPOTrainerNoKL(Trainer):


    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        score_funcs: Union[ScoreFunc, list[ScoreFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        score_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            model_init_kwargs["torch_dtype"] = torch.bfloat16
            if "Qwen2-VL" in model_id:
                model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Qwen2.5-VL" in model_id:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Aria" in model_id:
                model_init_kwargs.pop("use_cache")
                model = AriaForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )
        print(model)

        ### 冻结视觉层参数
        for param in model.visual.parameters():
            param.requires_grad = False

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        self.ref_model = None

        # Processing class
        if processing_class is None:
            if "Qwen2-VL" in model_id or "Qwen2.5-VL" in model_id or "Aria" in model_id:
                processing_class = AutoProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                if "Qwen" in model_id or "Qwen2.5-VL" in model_id:
                    processing_class.image_processor.max_pixels = max_pixels
                    processing_class.image_processor.min_pixels = min_pixels
            else:
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
                pad_token_id = processing_class.pad_token_id

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Score functions
        for i, score_func in enumerate(score_funcs):
            if isinstance(score_func, str):
                score_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    score_func, num_labels=1, **model_init_kwargs
                )
        self.score_funcs = score_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Score processing class
        if score_processing_classes is None:
            score_processing_classes = [None] * len(score_funcs)
        else:
            if len(score_processing_classes) != len(score_funcs):
                raise ValueError("The number of score processing classes must match the number of score functions.")
        self.score_processing_classes = score_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,  
            temperature=1, # HACK
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )
        self.beta = args.beta

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]


    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, **kwargs):
        # logits = model(input_ids, attention_mask=attention_mask, pixel_values=pixel_values, image_grid_thw=image_grid_thw).logits  # (B, L, V)
        logits = model(input_ids, **kwargs).logits
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)


    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def _crop_image_for_next_stage(self, output_text, origin_image, crop_bbox_to_cal_iou, input_width, input_height, width, height):

        img_with_bboxes = origin_image.copy()

        pattern = r"\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]"
        try:
            bbox = [list(map(float, match)) for match in re.findall(pattern, output_text[0])][0]
        except Exception as e:
            bbox = []
            print("ERROR:" + output_text[0])

        crop_bbox_to_cal_iou.append(cal_bbox_for_iou(bbox, input_width, input_height, width, height))

        if not bbox:
            print(f"no bbox:{output_text}")
        else:
            bbox = bbox_adjust(bbox, input_width, input_height, width, height, min_size=28)
            if bbox:
                # draw = ImageDraw.Draw(img_with_bboxes)
                # draw.rectangle(bbox, outline="red", width=2)
                img_with_bboxes = img_with_bboxes.crop(bbox)
            
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        # img_box_path = os.path.join("/map-vepfs/caomeng/code/MoBA/Video-CoT/output_r1/debug_img/", f"imagebox{i}_{timestamp}.png")
        # img_with_bboxes.save(img_box_path, "PNG")
        return img_with_bboxes

    def _get_bbox_for_last_stage(self, output_text, origin_image, crop_bbox_to_cal_iou, input_width, input_height, width, height):

        img_with_bboxes = origin_image.copy()

        pattern = r"\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]"
        try:
            bbox = [list(map(float, match)) for match in re.findall(pattern, output_text[0])][0]
        except Exception as e:
            bbox = []
            print("ERROR:" + output_text[0])

        crop_bbox_to_cal_iou.append(cal_bbox_for_iou(bbox, input_width, input_height, width, height))

    def _generate_for_stage2(self, origin_problem, input_width, input_height, combined_images, image_stage2, bbox):
        stage_two_prompt = {
            "prompt": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": STAGE_ONE_TEMPLATE.format(Question=origin_problem, input_width=input_width, input_height=input_height)},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": f"{bbox}"}]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": STAGE_TWO_TEMPLATE.format(Question=origin_problem, input_width=input_width, input_height=input_height)},
                ],
            },
            ]
        }
        _prompts = stage_two_prompt.get('prompt')
        prompts_text = self.processing_class.apply_chat_template(_prompts,tokenize=False, add_generation_prompt=True, add_vision_id=True)
        print(f"combined_images_intofunc---{combined_images}")
        combined_images.append(image_stage2)
        print(f"combined_images_outfuncs---{combined_images}")

        prompt_stage2_inputs = self.processing_class(
            text=prompts_text,
            images=combined_images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_stage2_inputs = super()._prepare_inputs(prompt_stage2_inputs)
        return prompt_stage2_inputs, _prompts

    def _generate_for_stage3(self, origin_problem, _prompts_stage2, input_width, input_height, combined_images, image_stage3, text_stage3):
        _prompts_stage2.append(
            {"role": "assistant", "content": [{"type": "text", "text": f"{text_stage3}"}]}
        )
        _prompts_stage2.append(
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": STAGE_THREE_TEMPLATE.format(Question=origin_problem, input_width=input_width, input_height=input_height)},
                ],
            }
        )

        # stage_three_prompt = {
        #     "prompt": [
        #     {
        #         "role": "user",
        #         "content": [
        #             {"type": "image"},
        #             {"type": "text", "text": STAGE_ONE_TEMPLATE.format(Question=origin_problem, input_width=input_width, input_height=input_height)},
        #         ],
        #     },
        #     {"role": "assistant", "content": [{"type": "text", "text": f"{text_stage2}"}]},
        #     {
        #         "role": "user",
        #         "content": [
        #             {"type": "image"},
        #             {"type": "text", "text": STAGE_TWO_TEMPLATE.format(Question=origin_problem, input_width=input_width, input_height=input_height)},
        #         ],
        #     },
        #     {"role": "assistant", "content": [{"type": "text", "text": f"{text_stage3}"}]},
        #     {
        #         "role": "user",
        #         "content": [
        #             {"type": "image"},
        #             {"type": "text", "text": STAGE_THREE_TEMPLATE.format(Question=origin_problem, input_width=input_width, input_height=input_height)},
        #         ],
        #     },
        #     ]
        # }
        # _prompts_stage2 = stage_three_prompt.get('prompt')
        prompts_text = self.processing_class.apply_chat_template(_prompts_stage2,tokenize=False, add_generation_prompt=True, add_vision_id=True)
        combined_images.append(image_stage3)

        prompt_stage3_inputs = self.processing_class(
            text=prompts_text,
            images=combined_images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_stage3_inputs = super()._prepare_inputs(prompt_stage3_inputs)
        return prompt_stage3_inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
    
        origin_problem = inputs[0]["problem"]
        prompts = [x["prompt"] for x in inputs]

        # print(inputs)
        # prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompts_text = self.processing_class.apply_chat_template(inputs[0]['prompt'],tokenize=False, add_generation_prompt=True, add_vision_id=True)
        if "image" in inputs[0]:
            images = []
            for (cur_idx, cur_input) in enumerate(inputs):
                copy_input = cur_input.copy()
                copy_input['prompt'][0]['content'][0]['image'] = inputs[cur_idx]["image"]
                images.append(process_vision_info(copy_input["prompt"])[0])
        # print(f"prompts_text---{prompts_text}")

        images = images[0]
        prompt_inputs = self.processing_class(
            text=prompts_text,
            images=images if "image" in inputs[0] else None,
            videos=None,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )

        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        # print(f"images:{images}")
        input_height = prompt_inputs['image_grid_thw'][0][1]*14
        input_width = prompt_inputs['image_grid_thw'][0][2]*14
        width, height = images[0].size

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]
        
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            # prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config)

            # Generate N times, each generate one with the temp_generation_config , stack the output_ids to prompt_completion_ids, pad the empty places with number 151613
            num_generations = self.generation_config.num_return_sequences
            temp_generation_config = copy.deepcopy(self.generation_config)
            temp_generation_config.num_return_sequences = 1


            origin_image = images[0]

            prompt_stage3_pixel_values = []
            prompt_stage3_image_grid_thw = []
            all_completions = []

            crop_bbox_to_cal_iou_stage1 = []
            crop_bbox_to_cal_iou_stage2 = []
            crop_bbox_to_cal_iou_stage3 = []

            for i in range(num_generations):  # -1 because we already have one generation

                with torch.no_grad():

                    combined_images = copy.deepcopy(images)
                    print(f"combined_images_oringin---{combined_images}")

                    completion_stage1 = unwrapped_model.generate(**prompt_inputs, generation_config=temp_generation_config)
                    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(prompt_inputs.input_ids, completion_stage1)]
                    output_text_stage1 =  self.processing_class.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

                    ### clean up
                    del generated_ids_trimmed
                    completion_stage1 = completion_stage1.cpu()
                    gc.collect(); torch.cuda.empty_cache()

                    img_with_bboxes_stage2 = self._crop_image_for_next_stage(output_text_stage1, images[0], crop_bbox_to_cal_iou_stage1, input_width, input_height, width, height)
                    prompt_stage2_inputs, _prompts_stage2 = self._generate_for_stage2(origin_problem, input_width, input_height, combined_images, img_with_bboxes_stage2, output_text_stage1[0])

                    completion_stage2 = unwrapped_model.generate(**prompt_stage2_inputs, generation_config=temp_generation_config)
                    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(prompt_stage2_inputs.input_ids, completion_stage2)]
                    output_text_stage2 =  self.processing_class.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

                    ### clean up
                    del generated_ids_trimmed
                    completion_stage2 = completion_stage2.cpu()
                    gc.collect(); torch.cuda.empty_cache()

                    img_with_bboxes_stage3 = self._crop_image_for_next_stage(output_text_stage2, images[0], crop_bbox_to_cal_iou_stage2, input_width, input_height, width, height)
                    prompt_stage3_inputs = self._generate_for_stage3(origin_problem, _prompts_stage2, input_width, input_height, combined_images, img_with_bboxes_stage3, output_text_stage2[0])

                    completion_stage3 = unwrapped_model.generate(**prompt_stage3_inputs, generation_config=temp_generation_config)
                    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(prompt_stage3_inputs.input_ids, completion_stage3)]
                    output_text_stage3 =  self.processing_class.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    self._get_bbox_for_last_stage(output_text_stage3, images[0], crop_bbox_to_cal_iou_stage3, input_width, input_height, width, height)

                    # all_completions.append(completion_stage3)
                    
                    # prompt_stage3_pixel_values.append(prompt_stage3_inputs["pixel_values"])
                    # prompt_stage3_image_grid_thw.append(prompt_stage3_inputs["image_grid_thw"])

                    # save results (on CPU)
                    all_completions.append(completion_stage3)
                    prompt_stage3_pixel_values.append(prompt_stage3_inputs["pixel_values"])
                    prompt_stage3_image_grid_thw.append(prompt_stage3_inputs["image_grid_thw"])
                    # clean up stage3
                    del completion_stage3, generated_ids_trimmed
                    gc.collect(); torch.cuda.empty_cache()

                # Log current memory usage
                print(f"[GPU] Allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB | Cached: {torch.cuda.memory_reserved() / 1e6:.1f} MB")
                
            # Stack all completions and pad if needed
            max_length = max(completion.size(1) for completion in all_completions)
            padded_completions = []
            indices_list = []

            for completion in all_completions:

                _seq = completion[0]
                start_pattern = torch.tensor([151644, 77091, 198], device=_seq.device)  # 保证同设备
                end_token = torch.tensor(151645, device=_seq.device)
                # print(f"seq--{_seq}")
                sample_indices = []
                start_indices = (_seq.unfold(0, len(start_pattern), 1) == start_pattern).all(dim=1).nonzero(as_tuple=True)[0]
                for start_index in start_indices:
                    start_index = start_index.item() + len(start_pattern)
                    end_indices = (_seq[start_index:] == end_token).nonzero(as_tuple=True)[0]
                    ## end_index要计算loss
                    # end_index = start_index + end_indices[0].item() + 1 # 有概率报错 IndexError: index 0 is out of bounds for dimension 0 with size 0
                    end_index = start_index + (end_indices[0].item() + 1 if end_indices.numel() > 0 else len(_seq) - start_index)
                    sample_indices.append(list(range(start_index, end_index)))
                indices_list.append(sample_indices)

                if completion.size(1) < max_length:
                    padding = torch.full(
                        (completion.size(0), max_length - completion.size(1)),
                        self.processing_class.tokenizer.pad_token_id,
                        dtype=completion.dtype,
                        device=completion.device,
                    )
                    padded_completion = torch.cat([completion, padding], dim=1)
                else:
                    padded_completion = completion
                padded_completions.append(padded_completion)
            
            # Stack all padded completions
            prompt_completion_ids = torch.cat(padded_completions, dim=0)
        # print(f"indices_list2---{indices_list}")

        
        completion_ids = prompt_completion_ids[:]  # shape: (batch_size, seq_length)

        device = self.accelerator.device
        batch_size, seq_length = completion_ids.size()
        completion_mask = torch.zeros((batch_size, seq_length), dtype=torch.int, device=device)

        # 3. 使用 indices_list 来设置 mask（保留 indices_list 内的 token）
        for i, indices in enumerate(indices_list):
            for index_range in indices:
                completion_mask[i, index_range] = 1  # 将 indices 范围内的 token 设为 1（不屏蔽）

        # print("completion_ids:", completion_ids.tolist())
        # print("completion_mask:", completion_mask.shape)

        prompt_stage3_inputs.pop("input_ids")
        prompt_stage3_inputs.pop("attention_mask")
        if "image" in inputs[0]:
            # prompt_stage2_inputs["pixel_values"] = prompt_stage2_inputs["pixel_values"].repeat(len(prompt_completion_ids), 1)
            # prompt_stage2_inputs["image_grid_thw"] = prompt_stage2_inputs["image_grid_thw"].repeat(len(prompt_completion_ids), 1)
            prompt_stage3_inputs["pixel_values"] = torch.cat(prompt_stage3_pixel_values, dim=0)
            prompt_stage3_inputs["image_grid_thw"] = torch.cat(prompt_stage3_image_grid_thw, dim=0)
        per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, **prompt_stage3_inputs)
        # print(f"per_token_logps1.shape---{per_token_logps.shape}")
        # print("completion_mask1:", completion_mask.shape)
        completion_mask = completion_mask[:, 1:]
        # per_token_logps = per_token_logps[:, prompt_length - 1 :]


        # Decode the generated completions
        ### 之前completion_ids做了截断，现在没有
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        # print(f"completions---{completions}")
        if is_conversational(inputs[0]):
            completions = [re.findall(r'(?<=\nassistant\n)(.*?)(?=\nuser\n|\Z)', completion, re.S) for completion in completions]
            # completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        # Compute the rewards
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:
                        # Repeat each value in the column for `num_generations` times
                        reward_kwargs[key].extend([example[key]] * self.num_generations)
                output_reward_func = reward_func(prompts=prompts, completions=completions, crop_bbox_to_cal_iou_stage3=crop_bbox_to_cal_iou_stage3, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Compute the score
        scores_per_func = torch.zeros(len(prompts), len(self.score_funcs), device=device)
        for i, (score_func, score_processing_class) in enumerate(
            zip(self.score_funcs, self.score_processing_classes)
        ):
            score_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
            for key in score_kwargs:
                for example in inputs:
                    score_kwargs[key].extend([example[key]] * self.num_generations)
            output_score_func = score_func(prompts=prompts, completions=completions, crop_bbox_to_cal_iou_stage1=crop_bbox_to_cal_iou_stage1, crop_bbox_to_cal_iou_stage2=crop_bbox_to_cal_iou_stage2, **score_kwargs)
            scores_per_func[:, i] = torch.tensor(output_score_func, dtype=torch.float32, device=device)


        # Sum the rewards from all reward functions
        rewards = rewards_per_func.sum(dim=1)
        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # x - x.detach() allows for preserving gradients from x
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -per_token_loss
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        # Log score function metric
        score_per_func = self.accelerator.gather_for_metrics(scores_per_func).mean(0)
        for i, score_func in enumerate(self.score_funcs):
            score_func_name = score_func.__name__
            self._metrics[f"scores/{score_func_name}"].append(score_per_func[i].item())


        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())

        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())
        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))

import torch
import torch.nn as nn
import deepspeed
from transformers import Trainer
from transformers.trainer_pt_utils import nested_detach
from transformers.utils import is_sagemaker_mp_enabled
from transformers.trainer import *
from transformers.integrations import is_deepspeed_zero3_enabled
import sys

class CPMTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        if "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        self.model.resampler.pos_embed = self.model.resampler.pos_embed.to(self.model.device)
        if is_deepspeed_zero3_enabled():
            with deepspeed.zero.GatheredParameters(self.model.resampler.attn.parameters(), modifier_rank=0):
                if not self.args.use_lora:
                    outputs = self.model(data = inputs, use_cache=False)
                else:
                    with self.model._enable_peft_forward_hooks(**inputs):
                        outputs = self.model.base_model(data = inputs, use_cache=False)
        else:
            if not self.args.use_lora:
                outputs = self.model(data = inputs, use_cache=False)
            else:
                with self.model._enable_peft_forward_hooks(**inputs):
                    outputs = self.model.base_model(data = inputs, use_cache=False)
                
        if labels is not None:
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            logits = outputs.logits.view(-1,
                                         self.model.config.vocab_size).contiguous()
            labels = labels.view(-1).long().contiguous()
            # Enable model parallelism
            labels = labels.to(logits.device)
            loss = loss_fct(logits, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = (
            False
            if len(self.label_names) == 0
            else all(inputs.get(k) is not None for k in self.label_names)
        )
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = (
            True if len(self.label_names) == 0 and return_loss else False
        )

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(
                    self.model.config, "keys_to_ignore_at_inference", []
                )
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name)
                                   for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels or loss_without_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(
                            v
                            for k, v in raw_outputs.items()
                            if k not in ignore_keys + ["loss"]
                        )
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(
                            v for k, v in raw_outputs.items() if k not in ignore_keys
                        )
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels or loss_without_labels:
                    with self.compute_loss_context_manager():
                        loss, outputs = self.compute_loss(
                            model, inputs, return_outputs=True
                        )
                    loss = loss.mean().detach()

                    if isinstance(outputs, dict):
                        logits = tuple(
                            v
                            for k, v in outputs.items()
                            if k not in ignore_keys + ["loss"]
                        )
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    with self.compute_loss_context_manager():
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(
                            v for k, v in outputs.items() if k not in ignore_keys
                        )
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)
        
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        del inputs
        torch.cuda.empty_cache()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            if is_deepspeed_zero3_enabled():
                with deepspeed.zero.GatheredParameters(self.model.resampler.attn.parameters(), modifier_rank=0):
                    self.accelerator.backward(loss)
            else:
                self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps
    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, supported_classes):
            if state_dict is None:
                state_dict = self.model.state_dict()

            if isinstance(unwrap_model(self.model), supported_classes):
                unwrap_model(self.model).save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if self.args.save_safetensors:
                    safetensors.torch.save_file(
                        state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
                    )
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            if self.args.use_lora:
                from collections import OrderedDict
                state_dict_vision = OrderedDict()
                for key, values in state_dict.items():
                    if 'vpm' in key or 'resampler' in key or 'embed_tokens' in key:
                        state_dict_vision[key] = values
                self.model.save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
                torch.save(state_dict_vision, f"{output_dir}/vpm_resampler_embedtokens.pt", )
            else:
                self.model.save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

import copy
import json
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoTokenizer

llama3_chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '' + message['role'] + '\n\n'+ message['content'] | trim + '' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}"

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        raw_data,
        transform,
        tokenizer,
        slice_config,
        llm_type="minicpm",
        patch_size=14,
        query_nums=64,
        batch_vision=False,
    ):
        super(SupervisedDataset, self).__init__()
        self.raw_data = raw_data
        self.tokenizer = tokenizer
        self.transform = transform
        self.slice_config = slice_config
        self.llm_type = llm_type
        self.patch_size = patch_size
        self.query_nums=query_nums
        self.batch_vision = batch_vision

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        image = Image.open(self.raw_data[i]["image"]).convert("RGB")
        ret = preprocess(
            image,
            self.raw_data[i]["conversations"],
            self.tokenizer,
            self.transform,
            query_nums=self.query_nums,
            slice_config=self.slice_config,
            llm_type=self.llm_type,
            patch_size=self.patch_size,
            batch_vision=self.batch_vision,
        )
        ret = dict(
            input_ids=ret["input_ids"],
            position_ids=ret["position_ids"],
            labels=ret["target"],
            attention_mask=torch.ones_like(ret["input_ids"], dtype=torch.bool),
            pixel_values=ret["pixel_values"],
            tgt_sizes=ret["tgt_sizes"],
            image_bound=ret["image_bound"],
        )

        return ret

def data_collator(examples, padding_value=0, max_length=2048):
    def trim_and_pad(seq, batch_first, padding_value):
        return pad_sequence([s[:max_length] for s in seq], batch_first=True, padding_value=padding_value)

    input_ids = trim_and_pad(
        [example["input_ids"] for example in examples],
        batch_first=True,
        padding_value=padding_value,
    )
    position_ids = trim_and_pad(
        [example["position_ids"] for example in examples],
        batch_first=True,
        padding_value=padding_value,
    )
    targets = trim_and_pad(
        [example["labels"] for example in examples],
        batch_first=True,
        padding_value=-100,
    )
    attention_mask = trim_and_pad(
        [example["attention_mask"] for example in examples],
        batch_first=True,
        padding_value=padding_value,
    )
    pixel_values = [example["pixel_values"] for example in examples]
    image_bound = [example["image_bound"] for example in examples]
    tgt_sizes = [example["tgt_sizes"] for example in examples]
    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "labels": targets,
        "attention_mask": attention_mask,
        "image_bound": image_bound,
        "tgt_sizes": tgt_sizes,
        "pixel_values": pixel_values,
    }


def conversation_to_ids(conversation, tokenizer, llm_type=None):
    """
    for single image multi-turn conversation
    conversation: [{'role': 'user', 'content': 'Describe this image'},
                   {'role': 'assistant', 'content': 'This is a cat.'}]
    """
    if llm_type == "llama3":
        input_ids, context, raw_msg = conversation_to_ids_llama3(
            conversation, tokenizer
        )
    else:
        input_ids, context, raw_msg = conversation_to_ids_minicpm(
            conversation, tokenizer
        )

    ids = torch.from_numpy(np.hstack(input_ids, dtype=np.int32))
    context = torch.from_numpy(np.hstack(context, dtype=np.int8))

    # build target
    target = torch.full_like(ids, -100, dtype=torch.int32)
    for i in range(1, len(ids)):
        if context[i] == 0:
            target[i - 1] = ids[i]
        if context[i] == 1 and context[i - 1] == 0:
            if hasattr(tokenizer, "eot_id"):
                target[i - 1] = tokenizer.eot_id
            else:
                target[i - 1] = tokenizer.eos_id

    # build image bound
    image_start_tokens = torch.where(ids == tokenizer.im_start_id)[0]
    image_start_tokens += 1
    image_end_tokens = torch.where(ids == tokenizer.im_end_id)[0]
    if len(image_start_tokens) != len(image_end_tokens):
        print("image start token != image end tokens")
        
    if len(image_start_tokens) > 0:
        image_bound = torch.hstack(
            [image_start_tokens.unsqueeze(-1), image_end_tokens.unsqueeze(-1)]
        )
    else:
        image_bound = []

    position_ids = torch.arange(ids.size(0)).long()
    return {
        "input_ids": ids,
        "target": target,
        "image_bound": image_bound,
        "raw_msg": raw_msg,
        "position_ids": position_ids
    }

def conversation_to_ids_minicpm(conversation, tokenizer):
    raw_msg = ""
    input_ids = []
    context = []
    for idx, msg in enumerate(conversation):
        role = msg["role"]
        message = msg["content"]
        assert role in ["user", "assistant"]
        if role == "user":
            prefix = "<用户>"
        else:
            prefix = "<AI>"
        # append eos
        if idx == len(conversation) - 1:
            message = message + tokenizer.eos_token
        prefix_ids = tokenizer.encode(prefix)[1:]  # remove bos
        message_ids = tokenizer.encode(message)[1:]

        input_ids.append(prefix_ids)
        input_ids.append(message_ids)

        context.append(np.ones((len(prefix_ids),), dtype=np.int8))
        if role == "assistant":
            context.append(np.zeros((len(message_ids),), dtype=np.int8))
        else:
            context.append(np.ones((len(message_ids),), dtype=np.int8))

        raw_msg += prefix + message

    return input_ids, context, raw_msg


def conversation_to_ids_llama3(conversation, tokenizer):
    raw_msg = ""
    input_ids = []
    context = []
    raw_msg = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False, chat_template=llama3_chat_template,
    )
    input_ids = tokenizer.apply_chat_template(
        conversation, tokenize=True, add_generation_prompt=False, chat_template=llama3_chat_template,
    )
    input_ids = np.array(input_ids)

    start_header_idxs = np.where(
        input_ids == tokenizer.convert_tokens_to_ids("")
    )[0]
    assistant_idxs = np.where(
        input_ids == tokenizer.convert_tokens_to_ids("assistant")
    )[0]
    end_header_idxs = np.where(
        input_ids == tokenizer.convert_tokens_to_ids("")
    )[0]
    eot_idxs = np.where(
        input_ids == tokenizer.convert_tokens_to_ids(""))[0]

    context = np.ones_like(input_ids, dtype=np.int8)

    for assistant_idx in assistant_idxs:
        if assistant_idx in set((start_header_idxs + end_header_idxs) / 2):
            st = assistant_idx + 3  # assistant\n\n
            for eot_idx in eot_idxs:
                if eot_idx > st:
                    context[st: eot_idx + 1] = 0
                    break

    input_ids = np.hstack(input_ids)
    context = np.hstack(context)

    return input_ids, context, raw_msg


def preprocess(
    image,
    conversation,
    tokenizer,
    transform,
    query_nums=64,
    slice_config=None,
    llm_type=None,
    patch_size=14,
    batch_vision=False,
):
    """
    single image preprocess, the image will be placed at the top of the conversation
    """
    conversation = copy.deepcopy(conversation)
    assert len(conversation) > 1, "conversation length must large than 2"
    assert conversation[0]["role"] == "user", "the first role must be user"

    if slice_config is not None:
        assert isinstance(slice_config, Dict)
        assert "patch_size" in slice_config
        assert "max_slice_nums" in slice_config
        assert "scale_resolution" in slice_config
    default_image_placeholder = (
        tokenizer.im_start + tokenizer.unk_token * query_nums + tokenizer.im_end
    )
    if slice_config:
        images = []
        source_image, patches, best_grid = slice_image(
            image,
            slice_config["max_slice_nums"],
            slice_config["scale_resolution"],
            slice_config["patch_size"],
        )
        images.append(source_image)
        image_placeholder = default_image_placeholder
        if len(patches) > 0:
            for i in range(len(patches)):
                for j in range(len(patches[0])):
                    images.append(patches[i][j])

            image_placeholder += get_grid_placeholder(
                tokenizer, best_grid, query_nums)
        images = [transform(i) for i in images]
    else:
        images = [transform(image)]
        image_placeholder = default_image_placeholder
    if "<image>" in conversation[0]["content"]:
        conversation[0]["content"] = conversation[0]["content"].replace(
            "<image>", image_placeholder
        )
    else:
        conversation[0]["content"] = (
            image_placeholder + "\n" + conversation[0]["content"]
        )

    input_dict = conversation_to_ids(conversation, tokenizer, llm_type)

    if batch_vision:
        tgt_sizes = []
        reshape_images = []
        for image in images:
            H, W = image.shape[1:]
            reshape_image = reshape_by_patch(image, patch_size)
            reshape_images.append(reshape_image)
            tgt_sizes.append([H // patch_size, W // patch_size])
        if tgt_sizes:
            tgt_sizes = torch.Tensor(tgt_sizes).type(torch.int32)

        input_dict["pixel_values"] = reshape_images
        input_dict["tgt_sizes"] = tgt_sizes

    else:
        input_dict["pixel_values"] = images
        input_dict["tgt_sizes"] = []

    return input_dict


def slice_image(
    image, max_slice_nums=9, scale_resolution=448, patch_size=14, never_split=False
):
    original_size = image.size
    original_width, original_height = original_size
    log_ratio = math.log(original_width / original_height)
    ratio = original_width * original_height / \
        (scale_resolution * scale_resolution)
    multiple = min(math.ceil(ratio), max_slice_nums)

    source_image = None
    best_grid = None
    patches = []

    if multiple <= 1 or never_split:
        # dont need to slice, upsample
        best_size = find_best_resize(
            original_size, scale_resolution, patch_size, allow_upscale=True
        )
        source_image = image.resize(best_size, Image.Resampling.BICUBIC)
    else:
        candidate_split_grids_nums = []
        for i in [multiple - 1, multiple, multiple + 1]:
            if i == 1 or i > max_slice_nums:
                continue
            candidate_split_grids_nums.append(i)

        # source image, down-sampling and ensure divided by patch_size
        best_resize = find_best_resize(
            original_size, scale_resolution, patch_size)
        source_image = image.copy().resize(best_resize, Image.Resampling.BICUBIC)
        candidate_grids = []

        # find best grid
        for split_grids_nums in candidate_split_grids_nums:
            m = 1
            while m <= split_grids_nums:
                if split_grids_nums % m == 0:
                    candidate_grids.append([m, split_grids_nums // m])
                m += 1

        best_grid = [1, 1]
        min_error = float("inf")
        for grid in candidate_grids:
            error = abs(log_ratio - math.log(grid[0] / grid[1]))
            if error < min_error:
                best_grid = grid
                min_error = error

        refine_size = get_refine_size(
            original_size, best_grid, scale_resolution, patch_size, allow_upscale=True
        )

        refine_image = image.resize(refine_size, Image.Resampling.BICUBIC)
        patches = split_to_patches(refine_image, best_grid)

    return source_image, patches, best_grid


def ensure_divide(length, patch_size):
    return max(round(length / patch_size) * patch_size, patch_size)


def find_best_resize(original_size, scale_resolution, patch_size, allow_upscale=False):
    width, height = original_size
    if (width * height > scale_resolution * scale_resolution) or allow_upscale:
        r = width / height
        height = int(scale_resolution / math.sqrt(r))
        width = int(height * r)
    best_width = ensure_divide(width, patch_size)
    best_height = ensure_divide(height, patch_size)
    return (best_width, best_height)


def get_refine_size(
    original_size, grid, scale_resolution, patch_size, allow_upscale=False
):
    width, height = original_size
    grid_x, grid_y = grid

    refine_width = ensure_divide(width, grid_x)
    refine_height = ensure_divide(height, grid_y)

    grid_width = refine_width / grid_x
    grid_height = refine_height / grid_y

    best_grid_size = find_best_resize(
        (grid_width, grid_height),
        scale_resolution,
        patch_size,
        allow_upscale=allow_upscale,
    )

    refine_size = (best_grid_size[0] * grid_x, best_grid_size[1] * grid_y)

    return refine_size


def split_to_patches(image, grid):
    patches = []
    width, height = image.size
    grid_x = int(width / grid[0])
    grid_y = int(height / grid[1])

    for i in range(0, height, grid_y):
        images = []
        for j in range(0, width, grid_x):
            box = (j, i, j + grid_x, i + grid_y)
            patch = image.crop(box)
            images.append(patch)
        patches.append(images)

    return patches


def get_grid_placeholder(tokenizer, grid, query_num):
    image_placeholder = (
        tokenizer.im_start + tokenizer.unk_token * query_num + tokenizer.im_end
    )

    cols = grid[0]
    rows = grid[1]
    slices = []
    for i in range(rows):
        lines = []
        for j in range(cols):
            lines.append(image_placeholder)
        slices.append("".join(lines))
    slice_placeholder = tokenizer.slice_start + \
        "\n".join(slices) + tokenizer.slice_end
    return slice_placeholder


def reshape_by_patch(image_tensor, patch_size):
    """
    :param image_tensor: shape [3, H, W]
    :param patch_size:
    :return: [3, patch_size, HW/patch_size]
    """
    patches = torch.nn.functional.unfold(
        image_tensor, (patch_size, patch_size), stride=(patch_size, patch_size)
    )

    patches = patches.reshape(image_tensor.size(0), patch_size, patch_size, -1)
    patches = patches.permute(0, 1, 3, 2).reshape(
        image_tensor.size(0), patch_size, -1)
    return patches

import glob
import json
import logging
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional, Union, Literal, Tuple
from types import MethodType
import torch
import transformers
from accelerate.utils import DistributedType
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

from transformers import AutoModel, AutoTokenizer
from transformers.integrations import deepspeed
from transformers import AutoModel, AutoTokenizer

# from dataset import SupervisedDataset, data_collator
# from trainer import CPMTrainer

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="openbmb/MiniCPM-V-2")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    tune_vision: Optional[bool] = field(default=True)
    tune_llm: Optional[bool] = field(default=True)
    llm_type: str = field(default="minicpm")
    use_lora: Optional[bool] = field(default=False)
    max_slice_nums: Optional[int] = field(default=9)


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: str = r"llm\..*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj)"
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False
    lora_modules_to_save: str = ""
    lora_layer_replication: Optional[List[Tuple[int, int]]] = None
    lora_layers_to_transform: Optional[List[int]] = None
    lora_layers_pattern: Optional[str] = None
   
def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


local_rank = None
def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer, output_dir: str, bias="none"):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        if trainer.args.use_lora:
            state_dict = get_peft_state_maybe_zero_3(
                trainer.model.named_parameters(), bias
            )
        else:
            state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    transform,
    data_collator=None,
    llm_type="minicpm",
    slice_config=None,
    patch_size=14,
    query_nums=64,
    batch_vision=False,
    max_length=256,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = SupervisedDataset

    rank0_print("Loading data...")

    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(
        train_json,
        transform,
        tokenizer,
        slice_config=slice_config,
        llm_type=llm_type,
        patch_size=patch_size,
        query_nums=query_nums,
        batch_vision=batch_vision,
    )

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(
            eval_json,
            transform,
            tokenizer,
            slice_config=slice_config,
            llm_type=llm_type,
            patch_size=patch_size,
            query_nums=query_nums,
            batch_vision=batch_vision,
        )
    else:
        eval_dataset = None

    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator= partial(data_collator, max_length=max_length),
    )


def get_parameter_number(model):
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
        
    return {'Total': all_param, 'Trainable': trainable_params}


local_rank = 0


def train():
    global local_rank
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    if '--config_file' in sys.argv:
        config_file_index = sys.argv.index('--config_file') + 1
        config_file_path = sys.argv[config_file_index]
        with open(config_file_path, 'r') as f:
            config = json.load(f)
        config_args = []
        for key, value in config.items():
            config_args.append(f"--{key}")
            config_args.append(str(value))
        sys.argv = sys.argv[:config_file_index-1] + config_args + sys.argv[config_file_index+1:]

    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()
    if training_args.use_lora:
        training_args.tune_llm = False
    if getattr(training_args, "deepspeed", None) : 
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    local_rank = training_args.local_rank
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    device_map = None
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning(
                "FSDP or ZeRO3 are not incompatible with QLoRA."
            )
    
    model = AutoModel.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=compute_dtype,
        device_map=device_map,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True
    )

    if not training_args.tune_vision:
        model.vpm.requires_grad_(False)
    if not training_args.tune_llm:
        model.llm.requires_grad_(False)
        
    if training_args.use_lora:
        if training_args.use_lora and training_args.tune_llm:
            raise ValueError("The model cannot simultaneously adjust LLM parameters and apply LoRA.")
            
        rank0_print("Currently using LoRA for fine-tuning the MiniCPM-V model.")
        for name, param in model.llm.named_parameters():
            param.requires_grad = False
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            layers_to_transform=lora_args.lora_layers_to_transform,
            task_type="CAUSAL_LM",
        )
        if not hasattr(model, 'get_input_embeddings'):
            def get_input_embeddings(self):
                return self.llm.get_input_embeddings()
            model.get_input_embeddings = MethodType(get_input_embeddings, model)
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )
        model = get_peft_model(model, lora_config)
        model.base_model.resampler.requires_grad_(True)
        model.base_model.llm.model.embed_tokens.weight.requires_grad_(True)
        if training_args.tune_vision:
            model.base_model.vpm.requires_grad_(True)
        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

    rank0_print(get_parameter_number(model))

    llm_type = training_args.llm_type    
    
    rank0_print(f'llm_type={llm_type}')

    
    # Load data
    if hasattr(model.config, "slice_config"):
        model.config.slice_config.max_slice_nums = training_args.max_slice_nums
        slice_config = model.config.slice_config.to_dict()
    else:
        model.config.max_slice_nums = training_args.max_slice_nums
        slice_config = model.config.to_dict()

    if hasattr(model.config, "batch_vision_input"):
        batch_vision = model.config.batch_vision_input
    else:
        batch_vision = False

    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        transform=model.transform,
        data_collator=data_collator,
        slice_config=slice_config,
        llm_type=llm_type,
        patch_size=model.config.patch_size,
        query_nums=model.config.query_num,
        batch_vision=batch_vision,
        max_length=training_args.model_max_length,
    )
    
    trainer = CPMTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )

    trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(
        trainer=trainer,
        output_dir=training_args.output_dir,
        bias=lora_args.lora_bias)


if __name__ == "__main__":
    train()


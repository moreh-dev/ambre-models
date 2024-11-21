import copy
import datetime
import importlib.metadata
import os
import sys
import time
from typing import Any, Mapping, Optional, Union

from accelerate.logging import get_logger
from datasets import load_dataset
from packaging import version
from peft import PeftModel
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset
from torch.utils.data import RandomSampler
from torch.utils.data import SequentialSampler
from tqdm.auto import tqdm
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import TrainerCallback
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES
from transformers.trainer_pt_utils import LabelSmoother
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available
from transformers.utils import is_peft_available
from trl import SFTTrainer

from moreh.driver.common import config as moreh_config

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

BAICHUAN_CHAT_TEMPLATE = "{% for message in messages %}{% if message['role'] == 'system' %}{{message['content']}}{% endif %}{% if message['role'] == 'user' %}{{'<reserved_106>' + message['content']}}{% endif %}{% if message['role'] == 'assistant' %}{{'<reserved_107>' + message['content']}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<reserved_107>' }}{% endif %}"

KEY = [
    'model_name_or_path', 'dataset_name_or_path', 'epochs', 'train_batch_size',
    'eval_batch_size', 'block_size', 'lr', 'use_lora'
]
LORA_KEY = ['lora_alpha', 'lora_dropout', 'lora_r']


def create_mask(input_ids, tokenizer):
    """
    Creates a mask with 1 for non-padding tokens and 0 for padding tokens.

    Args:
        input_ids (torch.Tensor): Tensor of input token IDs.
        tokenizer (PreTrainedTokenizer): Tokenizer with `pad_token_id`.

    Returns:
        torch.Tensor: Mask tensor (1 for valid tokens, 0 for padding).
    """

    pad_token_ids = (tokenizer.pad_token_id if tokenizer.pad_token_id
                     is not None else tokenizer.eos_token_id)
    return (input_ids != pad_token_ids).long()


def mask_pads(input_ids, attention_mask, ignore_index=-100):
    """
    Masks padding tokens in the input by setting them to `ignore_index`.

    Args:
        input_ids (torch.Tensor): Input token IDs.
        attention_mask (torch.Tensor): Mask indicating valid tokens (1) and padding (0).
        ignore_index (int, optional): Value to assign to padding tokens. Defaults to -100.

    Returns:
        torch.Tensor: Input token IDs with padding tokens replaced by `ignore_index`.
    """
    idx_mask = attention_mask
    labels = copy.deepcopy(input_ids)
    labels[~idx_mask.bool()] = ignore_index
    return labels


def load_model(args):
    """
    Loads a pre-trained model and tokenizer based on the model architecture specified in `args`.
    Optionally, LoRA (Low-Rank Adaptation) can be applied for model fine-tuning if specified
    in `args`.

    Args:
        args: Arguments containing the model path, LoRA settings, and other configurations.

    Returns:
        tuple: Loaded model and tokenizer.
    """

    print(f"Loading {args.model_name_or_path} Tokenizer...")
    set_mem_usage_correction_ratio(args)
    configs = AutoConfig.from_pretrained(args.model_name_or_path,
                                         trust_remote_code=True)
    if "baichuan" in configs.architectures[0].lower():
        from model.modeling_baichuan import BaichuanForCausalLM
        model = BaichuanForCausalLM.from_pretrained(args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                                  trust_remote_code=True)
    elif "llama" in configs.architectures[0].lower():
        from model.llama.modeling_llama import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                                  trust_remote_code=True)
    elif "qwen2" in configs.architectures[0].lower():
        from model.modeling_qwen2 import Qwen2ForCausalLM
        model = Qwen2ForCausalLM.from_pretrained(args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    elif "internlm" in configs.architectures[0].lower():
        from model.internlm.modeling_internlm2 import InternLM2ForCausalLM
        model = InternLM2ForCausalLM.from_pretrained(args.model_name_or_path,
                                                     trust_remote_code=True)
        model = convert_qkv_unfused(model)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                                  trust_remote_code=True)
    elif "qwen" in configs.architectures[0].lower():
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                     trust_remote_code=True,
                                                     torch_dtype='float32', fp32=True)
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-tokenizer",
                                                  trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if not tokenizer.pad_token:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if args.use_lora:
        from peft import get_peft_model
        from peft import LoraConfig
        if "baichuan" in configs.architectures[0].lower():
            _target_modules = ["W_pack"]
        elif "qwen" in configs.architectures[0].lower():
            _target_modules = ["c_proj"]
        else:
            _target_modules = ["q_proj", "v_proj"]
        config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            target_modules=_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        if any(name in configs.architectures[0].lower()
               for name in ["llama", "qwen2", "gemma"]):
            typecast_untrainable_params(model)
    print_trainable_parameters(model)
    return model, tokenizer


def load_custom_dataset(args):
    """
    Loads a dataset for training and validation based on the dataset path specified in `args`.

    Args:
        args: Arguments containing the dataset name or path.

    Returns:
        DatasetDict: A dictionary containing the loaded dataset with "train" and "validation" splits.
    """
    if args.dataset_name_or_path == "bitext/Bitext-customer-support-llm-chatbot-training-dataset":
        dataset = {}
        dataset["train"] = load_dataset(
            args.dataset_name_or_path, split="train[95%:]").with_format("torch")
        dataset["validation"] = load_dataset(
            args.dataset_name_or_path, split="train[:5%]").with_format("torch")
    elif args.dataset_name_or_path == "agileloop/izaz-sequence-of-actions-prediction-dataset-llama2-7b-32k":
        dataset = load_dataset(args.dataset_name_or_path).with_format("torch")
        dataset["train"] = load_dataset(
            args.dataset_name_or_path, split="train[:90%]").with_format("torch")
        dataset["validation"] = load_dataset(
            args.dataset_name_or_path,
            split="train[90%:95%]").with_format("torch")
    elif args.dataset_name_or_path == "alespalla/chatbot_instruction_prompts":
        dataset = load_dataset(args.dataset_name_or_path).with_format("torch")
        dataset["train"] = load_dataset(args.dataset_name_or_path,
                                        split="train[:90%]").with_format("torch")
        dataset["validation"] = load_dataset(
            args.dataset_name_or_path,
            split="train[90%:95%]").with_format("torch")
    else:
        dataset = load_dataset(args.dataset_name_or_path).with_format("torch")

    return dataset


def preprocess_dataset(args, dataset, tokenizer):
    """
    Preprocesses a dataset by tokenizing input prompts and applying a chat template for specified datasets.

    Different preprocessing methods are used based on the dataset, including handling the user-assistant chat format.
    Tokenization includes truncation, padding, and setting labels for training.

    Args:
        args: Arguments containing dataset configurations and block size for tokenization.
        dataset: The dataset to preprocess, containing training and validation splits.
        tokenizer: Tokenizer used for tokenizing the input data.

    Returns:
        DatasetDict: The preprocessed dataset with tokenized inputs and, optionally, labels.
    """

    def preprocess(prompt):
        chat = [
            {
                "role": "user",
                "content": f"{prompt['instruction']}"
            },
            {
                "role": "assistant",
                "content": f"{prompt['response']}"
            },
        ]
        chat = tokenizer.apply_chat_template(chat, tokenize=False)
        result = tokenizer(chat,
                           truncation=True,
                           max_length=args.block_size,
                           padding="max_length")
        result['labels'] = copy.deepcopy(result['input_ids'])
        return result

    def preprocess_chatbot(prompt):
        chat = [
            {
                "role": "user",
                "content": f"{prompt['prompt']}"
            },
            {
                "role": "assistant",
                "content": f"{prompt['response']}"
            },
        ]
        chat = tokenizer.apply_chat_template(chat, tokenize=False)
        chat += tokenizer.eos_token
        result = tokenizer(chat,
                           truncation=True,
                           max_length=args.block_size,
                           padding="max_length")
        result['labels'] = copy.deepcopy(result['input_ids'])
        return result

    def preprocess_agileloop(prompt):
        chat = [{
            "role": "user",
            "content": f"{prompt['Instruction']}"
        }, {
            "role": "assistant",
            "content": f"{prompt['Response']}"
        }]

        chat = tokenizer.apply_chat_template(chat, tokenize=False)
        result = tokenizer(chat,
                           truncation=True,
                           max_length=args.block_size,
                           padding="max_length")
        ret = {}
        result['labels'] = copy.deepcopy(result['input_ids'])
        ret['input_ids'] = result['input_ids']
        ret['attention_mask'] = result['attention_mask']
        ret['position_ids'] = torch.arange(0, len(result['labels']))  
        return ret

    if args.dataset_name_or_path == "bitext/Bitext-customer-support-llm-chatbot-training-dataset":
        dataset['train'] = dataset['train'].map(preprocess,
                                                num_proc=8,
                                                load_from_cache_file=True)
        dataset['validation'] = dataset['validation'].map(
            preprocess, num_proc=8, load_from_cache_file=True)
    elif args.dataset_name_or_path == "agileloop/izaz-sequence-of-actions-prediction-dataset-llama2-7b-32k":
        dataset['train'] = dataset['train'].map(preprocess_agileloop,
                                                num_proc=8,
                                                load_from_cache_file=True)
        dataset['validation'] = dataset['validation'].map(
            preprocess_agileloop, num_proc=8, load_from_cache_file=True)
    elif args.dataset_name_or_path == "MBZUAI/LaMini-instruction":
        dataset = dataset.map(preprocess, num_proc=1, load_from_cache_file=True)
    elif args.dataset_name_or_path == "alespalla/chatbot_instruction_prompts":
        dataset['train'] = dataset['train'].map(preprocess_chatbot,
                                                num_proc=1,
                                                load_from_cache_file=True)
        dataset['validation'] = dataset['validation'].map(
            preprocess_chatbot, num_proc=1, load_from_cache_file=True)
    else:
        dataset = dataset.map(preprocess, num_proc=8, load_from_cache_file=True)

    return dataset


def convert_qkv_unfused(model):
    """
    Converts a fused query, key, and value (QKV) weight matrix into separate Q, K, and V weight matrices
    for a model's attention layers, while ensuring gradients are frozen.

    This function is useful when models are trained with fused QKV matrices, but separated Q, K, and V
    matrices are needed for fine-tuning or inference.

    Args:
        model: The model with fused QKV matrices in its attention layers.

    Returns:
        model: The model with separate Q, K, and V matrices and frozen gradients for those weights.
    """
    config = model.config
    num_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads
    hidden_size = config.hidden_size
    num_key_value_groups = num_heads // num_key_value_heads
    head_dim = hidden_size // num_heads
    for name, module in model.named_modules():
        if name.split('.')[-1] != 'attention':
            continue
        wqkv = module.wqkv
        module.q.weight.requires_grad = False
        module.k.weight.requires_grad = False
        module.v.weight.requires_grad = False
        module.q.weight.copy_(
            wqkv.weight.view(
                num_key_value_heads, num_key_value_groups + 2, head_dim,
                hidden_size)[:, :num_key_value_groups, :, :].contiguous().view(
                    num_key_value_heads * num_key_value_groups * head_dim,
                    hidden_size))
        module.k.weight.copy_(
            wqkv.weight.view(num_key_value_heads, num_key_value_groups + 2,
                             head_dim,
                             hidden_size)[:, -2, :, :].contiguous().view(
                                 num_key_value_heads * head_dim, hidden_size))
        module.v.weight.copy_(
            wqkv.weight.view(num_key_value_heads, num_key_value_groups + 2,
                             head_dim,
                             hidden_size)[:, -1, :, :].contiguous().view(
                                 num_key_value_heads * head_dim, hidden_size))
        if config.bias:
            module.q.bias.requires_grad = False
            module.k.bias.requires_grad = False
            module.v.bias.requires_grad = False
            module.q.bias.copy_(
                wqkv.bias.view(
                    num_heads, num_key_value_groups + 2,
                    head_dim)[:, :num_key_value_groups, :].contiguous().view(
                        num_heads * num_key_value_groups * head_dim))
            module.k.bias.copy_(
                wqkv.bias.view(num_heads, num_key_value_groups + 2,
                               head_dim)[:, -2, :].contiguous().view(num_heads *
                                                                     head_dim))
            module.v.bias.copy_(
                wqkv.bias.view(num_heads, num_key_value_groups + 2,
                               head_dim)[:, -1, :].contiguous().view(num_heads *
                                                                     head_dim))
        del module.wqkv
    return model


def convert_qkv_fused(model):
    """
    Converts separate query, key, and value (Q, K, V) weight matrices into a fused QKV matrix
    for a model's attention layers, ensuring gradients are frozen.

    This function is useful for optimizing models that originally use separate Q, K, and V matrices
    by merging them into a single fused matrix.

    Args:
        model: The model with separate Q, K, and V matrices in its attention layers.

    Returns:
        model: The model with fused QKV matrices and frozen gradients for those weights.
    """
    config = model.config
    num_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads
    hidden_size = config.hidden_size
    num_key_value_groups = num_heads // num_key_value_heads
    head_dim = hidden_size // num_heads
    for name, module in model.named_modules():
        if name.split('.')[-1] != 'attention':
            continue
        wqkv = module.wqkv
        q = module.q
        k = module.k
        v = module.v
        wqkv.weight.requires_grad = False
        wqkv.weight.view(num_key_value_heads, num_key_value_groups + 2,
                         head_dim,
                         hidden_size)[:, :num_key_value_groups, :, :].copy_(
                             q.weight.view(num_key_value_heads,
                                           num_key_value_groups, head_dim,
                                           hidden_size))
        wqkv.weight.view(num_key_value_heads, num_key_value_groups + 2,
                         head_dim, hidden_size)[:, -2, :, :].copy_(
                             k.weight.view(num_key_value_heads, head_dim,
                                           hidden_size))
        wqkv.weight.view(num_key_value_heads, num_key_value_groups + 2,
                         head_dim, hidden_size)[:, -1, :, :].copy_(
                             v.weight.view(num_key_value_heads, head_dim,
                                           hidden_size))
        if config.bias:
            wqkv.bias.requires_grad = False
            wqkv.bias.view(num_key_value_heads, num_key_value_groups + 2,
                           head_dim)[:, :num_key_value_groups, :].copy(
                               q.bias.view(num_key_value_heads,
                                           num_key_value_groups, head_dim))
            wqkv.bias.view(num_key_value_heads, num_key_value_groups + 2,
                           head_dim)[:, -2, :].copy(
                               k.bias.view(num_key_value_heads, head_dim))
            wqkv.bias.view(num_key_value_heads, num_key_value_groups + 2,
                           head_dim)[:, -1, :].copy(
                               v.bias.view(num_key_value_heads, head_dim))
        del module.q
        del module.k
        del module.v
    return model


def save_model_and_tokenizer(args, model, tokenizer):
    """
    Saves the model and tokenizer to the specified path in `args`.

    For InternLM models, the function also converts separate Q, K, and V matrices into a fused QKV matrix
    before saving. The function saves both the model and tokenizer to the specified directory.

    Args:
        args: Arguments containing the save path for the model and tokenizer.
        model: The model to be saved.
        tokenizer: The tokenizer to be saved.

    Returns:
        None
    """
    config = model.config
    if "internlm" in config.architectures[0].lower():
        model.save_pretrained(args.save_path)
        from model.internlm.modeling_internlm2 import InternLM2ForCausalLM
        model = InternLM2ForCausalLM.from_pretrained(args.save_path,
                                                     trust_remote_code=True)
        model = convert_qkv_fused(model)

    print(f"Saving model and tokenizer in {args.save_path}")
    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)
    print(f"Model and Tokenizer is saved in {args.save_path}")


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model, along with the total number of parameters
    and the percentage of trainable parameters.

    Args:
        model: The model whose parameters are being evaluated.

    Returns:
        None
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def typecast_untrainable_params(model):
    """
    Converts the data type of untrainable (non-trainable) model parameters to `bfloat16`.

    This can help reduce memory usage for parameters that do not require gradient updates.

    Args:
        model: The model whose untrainable parameters will be typecast.

    Returns:
        None
    """
    for param in model.parameters():
        if not param.requires_grad:
            param.data = param.data.bfloat16()


def set_mem_usage_correction_ratio(args):
    """
    Sets the memory usage correction ratio for advanced parallelization if the attribute is present in `args`.

    This function adjusts memory management settings to optimize parallelization based on the specified ratio.

    Args:
        args: Arguments containing the memory usage correction ratio.

    Returns:
        None
    """
    if hasattr(args, "memory_usage_correction_ratio"):
        moreh_config.set_config(
            "advanced_parallelization_memory_usage_correction_ratio",
            args.memory_usage_correction_ratio)


class TrainCallback(TrainerCallback):

    def __init__(self, batch_size, world_size, warm_up_st, total_steps):
        self.duration_st = None
        self.duration_ed = None
        self.step_st = None
        self.warm_up_st = warm_up_st
        self.warm_up_ed = None
        self.eval_st = None
        self.eval_ed = None
        self.batch_size = batch_size
        self.tps = []
        self.step_tps = 0
        self.elapsed_times = []
        self.total_train_steps = total_steps
        self.world_size = world_size

    def on_train_begin(self, args, state, control, **kwargs):
        self.start = time.time()
        self.duration_st = time.time()
        self.accum = 0

    def on_step_begin(self, args, state, control, **kwargs):
        self.accum += 1

    def on_step_end(self, args, state, control, **kwargs):
        if (state.global_step % args.logging_steps == 0) or (state.global_step
                                                             == 1):
            control.should_log = True
        else:
            control.should_log = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step == 1:
            self.warmup_duration = time.time() - self.start
            self.start = time.time()
            self.accum = 0
        else:
            duration = time.time() - self.start
            tps = (args.max_seq_length * self.batch_size * self.accum *
                   self.world_size) / duration
            if 'loss' in logs:
                loss = logs['loss']
                lr = logs['learning_rate']
                if state.is_local_process_zero:
                    print(
                        f"[Step {state.global_step}] | TPS: {tps:.2f} tokens/sec | Loss: {loss:.6f} | LR : {lr:.8f} | Duration for 1 Step: {duration / self.accum:.2f} sec",
                        flush=True)
                self.tps.append(tps)
                self.elapsed_times.append(duration)
            self.accum = 0
            self.start = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        self.duration_ed = time.time()
        self.eval_st = time.time()

    def on_evaluate(self, args, state, control, **kwargs):
        self.eval_ed = time.time()

    def on_train_end(self, args, state, control, **kwargs):
        train_duration = self.duration_ed - self.duration_st
        warm_up_duration = self.warmup_duration
        if args.do_eval:
            eval_duration = self.eval_ed - self.eval_st
        else:
            eval_duration = 0
        avg_tps = sum(self.tps) / len(self.tps)
        avg_time_per_1_step = sum(self.elapsed_times) / (
            len(self.elapsed_times) * args.logging_steps - 1)
        total_steps = self.total_train_steps
        total_estimated_time = warm_up_duration + avg_time_per_1_step * (
            total_steps -
            1) + warm_up_duration + args.num_train_epochs * eval_duration
        days = total_estimated_time // 86400
        total_estimated_time -= days * 86400
        total_duration = train_duration + warm_up_duration + eval_duration
        print()
        print(f"{'Performance Summary':^40}")
        print("=" * 50)
        print(f"{'Total Duration:':<30} {total_duration:.2f} seconds")
        print(
            f"{'  Model Loading Duration:':<30} {warm_up_duration:.2f} seconds")
        print(f"{'  Train Duration:':<30} {train_duration:.2f} seconds")
        print(f"{'  Evaluation Duration:':<30} {eval_duration:.2f} seconds")
        print(
            f"{'Total Estimated Duration:':<30} {str(datetime.timedelta(days=days, seconds=total_estimated_time))} for {args.num_train_epochs} epochs"
        )
        print(f"{'Avg TPS:':<30} {avg_tps:.2f} tps")
        print("=" * 50)

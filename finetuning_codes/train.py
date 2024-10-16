import argparse
import copy
import time

from accelerate import Accelerator
from accelerate.logging import get_logger
import datasets
from datasets import load_dataset
from peft import get_peft_model
from peft import LoraConfig
import torch
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from tqdm.auto import tqdm
import transformers
from transformers import AutoTokenizer
from trl import SFTConfig
from trl import SFTTrainer
from utils import *


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="/root/poc/pretrained_models/Meta-Llama-3-70B-Instruct")
    parser.add_argument(
        "--dataset-name-or-path",
        type=str,
        default="bitext/Bitext-customer-support-llm-chatbot-training-dataset")
    parser.add_argument("--block-size", type=int, default=32768)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--output-dir",
                        type=str,
                        default="/root/poc/checkpoints/llama_lora_finetuned")
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--flash-attn", action="store_true")
    parser.add_argument("--log-interval", type=int, default=5)
    args = parser.parse_args()
    return args


def main(args):
    torch.moreh.option.enable_advanced_parallelization()

    accelerator = Accelerator()
    world_size = accelerator.num_processes
    logger = get_logger(__name__)
    logger.info(accelerator.state, main_process_only=True)
    logger.warning(accelerator.state, main_process_only=True)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                              trust_remote_code=True)
    dataset = load_custom_dataset(args)
    dataset = preprocess_dataset(args, dataset, tokenizer)

    # SFTConfig
    trainer_config = SFTConfig(
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        output_dir=args.output_dir,
        max_seq_length=1024,
        optim='adamw_torch',
        lr_scheduler_type="cosine",
        learning_rate=args.lr,
        warmup_steps=50,
        bf16=True,
        do_eval=True,
        eval_strategy="epoch",
        logging_steps=args.log_interval,
        report_to='none',
        logging_nan_inf_filter=False,
        save_strategy="no",
        max_grad_norm=0,
    )

    warm_up_st = time.time()

    total_train_steps = (len(dataset["train"]) //
                         (world_size * args.train_batch_size)) * args.num_epochs

    trainer = SFTTrainer(model,
                         tokenizer=tokenizer,
                         args=trainer_config,
                         train_dataset=dataset['train'],
                         eval_dataset=dataset['validation'],
                         callbacks=[
                             TrainCallback(
                                 batch_size=args.train_batch_size,
                                 world_size=world_size,
                                 warm_up_st=warm_up_st,
                                 total_steps=total_train_steps,
                             )
                         ])
    trainer.train()
    trainer.save_state()
    unwrapped_model = accelerator.unwrap_model(model)
    save_model_and_tokenizer(args, unwrapped_model, tokenizer)


if __name__ == "__main__":
    args = arg_parse()
    main(args)

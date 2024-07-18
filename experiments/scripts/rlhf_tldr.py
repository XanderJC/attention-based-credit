import torch
import time
import wandb
import datetime
import argparse
import logging
import os
import numpy as np
import re
from pkg_resources import resource_filename
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    set_seed,
)
from peft import LoraConfig, AutoPeftModelForSequenceClassification
from datasets import load_dataset
import sys

from abcrl.rl.ppo import PPOTrainerABC
from abcrl.attention.redistribution import (
    get_attention_distribution,
    get_generator_attention_distribution,
)
from trl import (
    PPOConfig,
    AutoModelForCausalLMWithValueHead,
)
from trl.core import LengthSampler


def build_tldr_dataset(
    config,
    dataset_name="openai/summarize_from_feedback",
):
    """
    Build dataset for training.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    ds = load_dataset(dataset_name, "comparisons", split="train")

    def tokenize(sample):
        choice, post, sum1, sum2 = (
            sample["choice"],
            sample["info"]["post"],
            sample["summaries"][0]["text"],
            sample["summaries"][1]["text"],
        )
        query = f"### Text to Summarize: {post}\n ### Summary: "
        sample["input_ids"] = tokenizer.encode(query)
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds = ds.filter(lambda x: len(x["input_ids"]) < 450, batched=False)
    ds.set_format(type="torch")
    return ds


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def main(
    method: str = "rlhf",
    max_epochs: int = 50,
    beta: float = 0.5,
    l_rate: float = 1.41e-5,
    min_generation: int = 8,
    max_generation: int = 16,
    project_name: str = "rlhf-tldr",
    batch_size: int = 16,
    seed: int = 41310,
    lora_rank: int = 32,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    logging_level: str = "DEBUG",
):
    assert method in ["rlhf", "abc", "abcde", "abcde2", "uniform"]

    now = datetime.datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M")

    run_name = f"{method}_{int(beta*100)}_{min_generation}_{max_generation}_{date_time}"

    print(f"Run name: {run_name}")
    BASE_PATH = resource_filename("abcrl", "/..")

    logger = logging.getLogger(__name__)
    level = logging.getLevelName(logging_level)
    logger.setLevel(level)
    LOG_DIRECTORY = f"{BASE_PATH}/experiments/logs/{run_name}"

    if not os.path.exists(LOG_DIRECTORY):
        os.makedirs(LOG_DIRECTORY)

    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(LOG_DIRECTORY + "/debug.log")
    c_handler.setLevel(logging.WARNING)
    f_handler.setLevel(level)

    c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    if seed is not None:
        print(f"Setting seed to {seed}")
        set_seed(seed)

    config = PPOConfig(
        model_name="XanderJC/gptj-sft-tldr-merged",
        learning_rate=l_rate,
        log_with="wandb",
        ppo_epochs=4,
        batch_size=batch_size,
        optimize_device_cache=True,
        seed=seed,
    )

    dataset = build_tldr_dataset(config)
    dataset = dataset.shuffle()
    logger.info(f"Dataset length: {len(dataset)}")
    logger.debug(dataset[0])

    wandb.init(**{"project": project_name, "name": run_name, "entity": "alex-abc"})

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model_name,
        peft_config=lora_config,
        quantization_config=nf4_config,
        trust_remote_code=True,
    ).to("cuda:0")
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model_name,
        quantization_config=nf4_config,
        trust_remote_code=True,
    ).to("cuda:0")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    rank_model = AutoPeftModelForSequenceClassification.from_pretrained(
        "Holarissun/trl_rm_tldr_gptj",
        num_labels=1,
        output_attentions=True,
        return_dict_in_generate=True,
        attn_implementation="eager",
    ).to("cuda:0")
    print(type(rank_model))
    rank_tokenizer = AutoTokenizer.from_pretrained("Holarissun/trl_rm_tldr_gptj")
    rank_tokenizer.pad_token = rank_tokenizer.eos_token
    rank_model.config.pad_token_id = rank_model.config.eos_token_id

    logging.getLogger().handlers[0].setLevel(logging.WARNING)

    ppo_trainer = PPOTrainerABC(
        config=config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
    )

    logger.info(f"PPO trainer device: {ppo_trainer.accelerator.device}")
    logger.info(f"PPO lpi: {ppo_trainer.accelerator.local_process_index}")
    logger.info(f"RM device: {rank_model.device}")

    print(f"PPO trainer device: {ppo_trainer.accelerator.device}")
    print(f"PPO lpi: {ppo_trainer.accelerator.local_process_index}")

    output_min_length = min_generation
    output_max_length = max_generation
    output_length_sampler = LengthSampler(output_min_length, output_max_length)

    local_res = []
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "return_dict_in_generate": True,
        "batch_size": 8,
    }

    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]
        start = time.time()
        #### Get response from gpt2
        response_tensors = []
        response_attentions = []
        for query in query_tensors:
            gen_len = max_generation
            generation_kwargs["max_new_tokens"] = gen_len
            generation_kwargs["min_new_tokens"] = min_generation
            response = ppo_trainer.generate(query, **generation_kwargs)

            response_tensors.append(response[0].squeeze()[len(query) :])
            response_attentions.append(response.attentions)

        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
        logger.debug(f'First response: {batch["response"][0]}')
        end = time.time()
        logger.info(f"Generation time: {round(end - start,1)}s")
        #### Compute sentiment score
        with torch.no_grad():
            start = time.time()
            texts = [q + r for q, r in zip(batch["query"], batch["response"])]
            print(
                len(batch["query"][0]) / 4,
                len(batch["response"][0]) / 4,
                len(texts[0]) / 4,
            )
            print("example case now: \n\n\n", texts[0])

            inputs = rank_tokenizer(
                texts,
                return_tensors="pt",
                max_length=512,
                padding="max_length",
                truncation=True,
            ).to("cuda:0")

            out = rank_model(**inputs)
        end = time.time()
        logger.info(f"RM Inference time: {round(end - start,1)}s")
        start = time.time()
        rewards = []

        if method == "abc":
            attention = out.attentions[-1].mean(1)  # last layer averaged over heads
        elif method == "abcde" or method == "abcde2":
            attention = response_attentions
        else:
            attention = [None] * len(response_tensors)

        for out, response, query, attention in zip(
            out.logits, response_tensors, query_tensors, attention
        ):
            total = out.detach()
            reward = torch.zeros_like(response, dtype=float)

            if method == "rlhf":
                reward[-1] = total

            elif method == "abc":
                reward[-1] = (1 - beta) * total
                redist_reward = (
                    torch.tensor(
                        get_attention_distribution(response, query, attention.cpu()),
                        device=reward.device,
                    )
                    * total.to(reward.device)
                    * beta
                )
                reward += redist_reward

            elif method == "abcde":
                reward[-1] = (1 - beta) * total
                redist_reward = (
                    torch.tensor(
                        get_generator_attention_distribution(
                            response, query, attention, False
                        ),
                        device=reward.device,
                    )
                    * total
                    * beta
                )
                reward += redist_reward

            elif method == "abcde2":
                reward[-1] = (1 - beta) * total
                redist_reward = (
                    torch.tensor(
                        get_generator_attention_distribution(
                            response, query, attention, True
                        ),
                        device=reward.device,
                    )
                    * total
                    * beta
                )
                reward += redist_reward

            elif method == "uniform":
                reward += total / len(reward)
            rewards.append(reward)
        end = time.time()
        logger.info(f"Reward calculation time: {round(end - start,1)}s")
        #### Run PPO step

        start = time.time()
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        end = time.time()
        logger.info(f"PPO step time: {round(end - start,1)}s")
        logger.debug(f"First reward: {rewards[0]}")

        og_rewards = [score.cpu().sum() for score in rewards]
        logger.debug(f"Total rewards: {og_rewards}")
        ppo_trainer.log_stats(stats, batch, og_rewards)

        stats["env/reward_mean"] = np.mean(og_rewards)
        stats["env/reward_std"] = np.std(og_rewards)
        del stats["objective/logprobs"]
        del stats["objective/ref_logprobs"]
        del stats["ppo/policy/advantages"]
        del stats["ppo/policy/ratio"]

        local_res.append(stats)
        torch.save(local_res, LOG_DIRECTORY + "/local_res.th")

        if epoch >= max_epochs:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="rlhf")
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--beta", type=float, default=0.8)
    parser.add_argument("--l_rate", type=float, default=1.41e-6)
    parser.add_argument("--min_generation", type=int, default=8)
    parser.add_argument("--max_generation", type=int, default=48)
    parser.add_argument("--project_name", type=str, default="rlhf-tldr-gptj")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=41310)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)

    args = parser.parse_args()

    main(**args.__dict__)

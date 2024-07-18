import torch
import wandb
import datetime
import argparse
import logging
import os
import numpy as np
import time
from pkg_resources import resource_filename
from tqdm import tqdm
from transformers import (
    LlamaTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    set_seed,
)
from peft import LoraConfig
from abcrl.rl.ppo import PPOTrainerABC
from abcrl.attention.redistribution import (
    get_attention_distribution,
    get_generator_attention_distribution,
)
from abcrl.datasets import build_anthropic_dataset, collator
from trl import (
    PPOConfig,
    AutoModelForCausalLMWithValueHead,
)


def main(
    method: str = "rlhf",
    max_epochs: int = 50,
    beta: float = 0.5,
    l_rate: float = 1.41e-5,
    lora_rank: int = 32,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    min_generation: int = 8,
    max_generation: int = 16,
    ppo_epochs: int = 10,
    use_score_scaling: bool = False,
    use_score_nomalization: bool = False,
    repetition_penalty: float = 1.0,
    max_instruction_length: int = 256,
    project_name: str = "rlhf",
    batch_size: int = 16,
    mini_batch_size: int = 1,
    seed: int = 41310,
    logging_level: str = "DEBUG",
):
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

    assert method in ["rlhf", "abc", "abcde", "abcde2", "uniform"]

    if seed is not None:
        print(f"Setting seed to {seed}")
        set_seed(seed)

    config = PPOConfig(
        model_name="VMware/open-llama-7b-open-instruct",
        learning_rate=l_rate,
        log_with="wandb",
        ppo_epochs=ppo_epochs,
        batch_size=batch_size,
        optimize_device_cache=True,
        remove_unused_columns=False,
        mini_batch_size=mini_batch_size,
        use_score_scaling=use_score_scaling,
        use_score_norm=use_score_nomalization,
    )
    logger.info(f"PPO Config: {config}")

    dataset = build_anthropic_dataset(config, max_instruction_length)
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
    )
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model_name, quantization_config=nf4_config
    )
    tokenizer = LlamaTokenizer.from_pretrained(config.model_name, use_fast=False)

    rank_model = AutoModelForSequenceClassification.from_pretrained(
        "weqweasdas/hh_rlhf_rm_open_llama_3b",
        output_attentions=True,
        return_dict_in_generate=True,
        attn_implementation="eager",
        device_map="cuda:0",
    )
    rank_tokenizer = LlamaTokenizer.from_pretrained(
        "weqweasdas/hh_rlhf_rm_open_llama_3b", use_fast=False
    )

    # need to reset stream handler after loading hf models
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

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "return_dict_in_generate": True,
        "batch_size": batch_size,
        "pad_token_id": tokenizer.pad_token_id,
        "repetition_penalty": repetition_penalty,
        "max_new_tokens": max_generation,
        "min_new_tokens": min_generation,
    }

    local_res = []

    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]
        rm_query_tensors = batch["rm_input_ids"]

        response_tensors = []
        response_attentions = []
        start = time.time()
        for query in query_tensors:
            response = ppo_trainer.generate(query, **generation_kwargs)
            response_tensors.append(response[0].squeeze()[len(query) :])
            response_attentions.append(response.attentions)

        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
        logger.debug(f'First response: {batch["response"][0]}')
        end = time.time()
        logger.info(f"Generation time: {round(end - start,1)}s")

        #### Compute Reward
        start = time.time()
        with torch.no_grad():
            texts = [q + r for q, r in zip(batch["rm_query"], batch["response"])]
            logger.debug(f"First RM input: {texts[0]}")
            inputs = rank_tokenizer(
                texts,
                return_tensors="pt",
                max_length=max_instruction_length + max_generation,
                padding="max_length",
                truncation=True,
            ).to(rank_model.device)
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

        for out, response, query, attention, rm_query in zip(
            out.logits, response_tensors, query_tensors, attention, rm_query_tensors
        ):
            total = out.detach()
            reward = torch.zeros_like(response, dtype=float)

            if method == "rlhf":
                reward[-1] = total

            elif method == "abc":
                reward[-1] = (1 - beta) * total
                redist_reward = (
                    torch.tensor(
                        get_attention_distribution(response, rm_query, attention),
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

        if epoch % 100 == 0:
            ppo_trainer.save_pretrained(
                f"{BASE_PATH}/experiments/saved_models/{run_name}"
            )

        if epoch >= max_epochs:
            break

    ppo_trainer.save_pretrained(f"{BASE_PATH}/experiments/saved_models/{run_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="abc")
    parser.add_argument("--max_epochs", type=int, default=2)
    parser.add_argument("--beta", type=float, default=0.8)
    parser.add_argument("--l_rate", type=float, default=3e-5)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--min_generation", type=int, default=8)
    parser.add_argument("--max_generation", type=int, default=256)
    parser.add_argument("--ppo_epochs", type=int, default=10)
    parser.add_argument("--use_score_scaling", type=bool, default=False)
    parser.add_argument("--use_score_nomalization", type=bool, default=False)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max_instruction_length", type=int, default=256)
    parser.add_argument("--project_name", type=str, default="openllama")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--mini_batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=41311)
    parser.add_argument("--logging_level", type=str, default="DEBUG")

    args = parser.parse_args()

    main(**args.__dict__)

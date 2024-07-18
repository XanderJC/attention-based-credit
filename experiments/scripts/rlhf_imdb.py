import torch
import wandb
import datetime
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, set_seed
from datasets import load_dataset
from abcrl.rl.ppo import PPOTrainerABC
from abcrl.attention.redistribution import (
    get_attention_distribution,
    get_generator_attention_distribution,
)
from trl import PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler


def build_dataset(
    config, dataset_name="imdb", input_min_text_length=2, input_max_text_length=8
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
    tokenizer.pad_token = tokenizer.eos_token
    # load imdb with datasets
    ds = load_dataset(dataset_name, split="train")
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def main(
    method: str = "abc",
    max_epochs: int = 50,
    beta: float = 0.5,
    l_rate: float = 1.41e-5,
    min_generation: int = 8,
    max_generation: int = 16,
    project_name: str = "rlhf",
    batch_size: int = 16,
    seed: int = 41310,
):
    assert method in ["rlhf", "abc", "abcde", "abcde2", "uniform"]

    if seed is not None:
        print(f"Setting seed to {seed}")
        set_seed(seed)

    now = datetime.datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M")

    run_name = f"{method}_{int(beta*100)}_{min_generation}_{max_generation}_{date_time}"

    print(f"Run name: {run_name}")

    wandb.init(**{"project": project_name, "name": run_name, "entity": "alex-abc"})

    config = PPOConfig(
        model_name="lvwerra/gpt2-imdb",
        learning_rate=l_rate,
        log_with="wandb",
        ppo_epochs=4,
        batch_size=batch_size,
    )

    dataset = build_dataset(config)
    dataset = dataset.shuffle()

    model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    tokenizer.pad_token = tokenizer.eos_token

    reward_name = "XanderJC/gpt2-rm-imdb"
    rank_model, rank_tokenizer = AutoModelForSequenceClassification.from_pretrained(
        reward_name, output_attentions=True
    ), AutoTokenizer.from_pretrained(reward_name)
    rank_model.config.pad_token_id = rank_model.config.eos_token_id

    ppo_trainer = PPOTrainerABC(
        config=config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
    )

    output_min_length = min_generation
    output_max_length = max_generation
    output_length_sampler = LengthSampler(output_min_length, output_max_length)

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "return_dict_in_generate": True,
        "output_attentions": True,
    }

    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]

        #### Get response from gpt2
        response_tensors = []
        response_attentions = []
        for query in query_tensors:
            gen_len = output_length_sampler()
            generation_kwargs["max_new_tokens"] = gen_len
            generation_kwargs["min_new_tokens"] = min_generation
            response = ppo_trainer.generate(query, **generation_kwargs)
            response_tensors.append(response[0].squeeze()[-gen_len:])
            response_attentions.append(response.attentions)

        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

        #### Compute sentiment score
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]

        inputs = rank_tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        )
        out = rank_model(**inputs)

        rewards = []
        if method == "abc":
            attention = out.attentions[-1].mean(1)  # last layer averaged over heads
        elif method == "abcde" or method == "abcde2":
            attention = response_attentions
        else:
            attention = [None] * len(response_tensors)

        for logit, response, query, attention in zip(
            out.logits, response_tensors, query_tensors, attention
        ):
            total = (logit[1] - logit[0]).detach()
            reward = torch.zeros_like(response, dtype=float)

            if method == "rlhf":
                reward[-1] = total

            elif method == "abc":
                reward[-1] = (1 - beta) * total
                redist_reward = (
                    torch.tensor(
                        get_attention_distribution(response, query, attention),
                        device=reward.device,
                    )
                    * total
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

        #### Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        og_rewards = [score.sum() for score in rewards]
        ppo_trainer.log_stats(stats, batch, og_rewards)

        if epoch >= max_epochs:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="abc")
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--l_rate", type=float, default=1.41e-5)
    parser.add_argument("--min_generation", type=int, default=8)
    parser.add_argument("--max_generation", type=int, default=16)
    parser.add_argument("--project_name", type=str, default="rlhf")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    main(**args.__dict__)

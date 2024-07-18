import os
import torch
from torch import nn
import math
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import snapshot_download

"""
Code mostly from https://huggingface.co/berkeley-nest/Starling-RM-7B-alpha
"""
## Define the reward model function class


class GPTRewardModel(nn.Module):
    def __init__(self, model_path, use_quantization=False, device="auto"):
        super().__init__()
        if not use_quantization:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, attn_implementation="eager"
            )
        else:
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path, quantization_config=nf4_config
            )
        self.config = model.config
        self.config.n_embd = (
            self.config.hidden_size
            if hasattr(self.config, "hidden_size")
            else self.config.n_embd
        )
        self.model = model
        self.transformer = model.model
        self.v_head = nn.Linear(
            self.config.n_embd, 1, bias=False
        )  # .to(self.model.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.PAD_ID = self.tokenizer(self.tokenizer.pad_token)["input_ids"][0]

    def get_device(self):
        return self.model.device

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
    ):
        """
        input_ids, attention_mask: torch.Size([bs, seq_len])
        return: scores: List[bs]
        """
        bs = input_ids.shape[0]
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        # print(transformer_outputs.attentions[-1].shape)
        # print(transformer_outputs.attentions.shape)
        hidden_states = transformer_outputs[0]
        scores = []
        rewards = self.v_head(hidden_states).squeeze(-1)
        for i in range(bs):
            c_inds = (input_ids[i] == self.PAD_ID).nonzero()
            c_ind = c_inds[0].item() if len(c_inds) > 0 else input_ids.shape[1]
            scores.append(rewards[i, c_ind - 1])
        return scores, transformer_outputs.attentions

    def load_model(self, checkpoint):
        directory = snapshot_download(checkpoint)

        for fpath in os.listdir(directory):
            if fpath.endswith(".pt") or fpath.endswith("model.bin"):
                checkpoint = os.path.join(directory, fpath)
                break

        self.load_state_dict(torch.load(checkpoint), strict=False)
        self.eval().requires_grad_(False)


def get_reward(
    samples, reward_model, reward_tokenizer, reward_device, reward_batch_size
):
    """samples: List[str]"""
    input_ids = []
    attention_masks = []
    encodings_dict = reward_tokenizer(
        samples,
        truncation=True,
        max_length=2048,
        padding="max_length",
        return_tensors="pt",
    ).to(reward_device)
    input_ids = encodings_dict["input_ids"]
    attention_masks = encodings_dict["attention_mask"]
    mbs = reward_batch_size
    out = []
    for i in range(math.ceil(len(samples) / mbs)):
        rewards = reward_model(
            input_ids=input_ids[i * mbs : (i + 1) * mbs],
            attention_mask=attention_masks[i * mbs : (i + 1) * mbs],
        )
        out.extend(rewards)
    return torch.hstack(out)


if __name__ == "__main__":
    ## Load the model and tokenizer
    import time

    reward_model = GPTRewardModel("meta-llama/Llama-2-7b-chat-hf")
    reward_tokenizer = reward_model.tokenizer
    reward_tokenizer.truncation_side = "left"
    reward_model.load_model("berkeley-nest/Starling-RM-7B-alpha")

    reward_model.model.config.output_attentions = True
    reward_model.model.config.return_dict_in_generate = True

    ## Define the reward function

    reward_device = reward_model.get_device()
    reward_batch_size = 8

    ## Inference over test prompts with llama2 chat template

    texts = [
        "<s>[INST] Hello? </s> [/INST] Hi, how can I help you?</s>",
        "\n\nHuman: 0.002 = 1000 \n1 = x?\n\nAssistant: To find the value of x, we can set up a proportion using the given information:\n\n0.002/1000 = 1/x\n\nTo solve for x, we can cross multiply:\n\n0.002 * x = 1000 * 1\n\n0.002x = 1000\n\nDividing both sides by 0.002:\n\nx = 1000 / 0.002\n\nx = 500,000\n\nTherefore, 1 is equal to 500,000 in this proportion.",
    ]

    inputs = reward_tokenizer(
        texts[0],
        return_tensors="pt",
        max_length=2048,
        padding="max_length",
        truncation=True,
    )  # .to(reward_device)

    print(reward_device)
    start = time.time()
    out, attentions = reward_model(**inputs)
    end = time.time()
    print(f"Time: {end - start}")
    print(attentions[-1].shape)
    print(out)
    exit()

    reward_for_test_sample = get_reward(
        texts, reward_model, reward_tokenizer, reward_device, reward_batch_size
    )
    print(reward_for_test_sample)

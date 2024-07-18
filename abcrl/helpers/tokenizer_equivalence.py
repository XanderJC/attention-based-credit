from transformers import AutoTokenizer, LlamaTokenizer


def check_special_tokens(tokenizer1, tokenizer2):
    """
    Compare the special tokens of two tokenizers and check if they are the same.

    Args:
        tokenizer1: The first tokenizer object.
        tokenizer2: The second tokenizer object.

    Returns:
        None: This function does not return anything.
    """
    special_tokens1 = tokenizer1.all_special_tokens
    special_tokens2 = tokenizer2.all_special_tokens

    print("Special Tokens:")
    different_tokens = []
    for token1, token2 in zip(special_tokens1, special_tokens2):
        if token1 != token2:
            different_tokens.append((token1, token2))

    same = True
    if len(different_tokens) > 0:
        print("Special tokens are different:")
        for token1, token2 in different_tokens:
            print(f"Token1: {token1}, Token2: {token2}")
        same = False
    else:
        print("Special tokens are the same.")
    return same


def test_tokenizer_equivalence(model1, model2, texts=None, verbose=False):
    """
    Compare the tokenization results of two models or tokenizers.

    Args:
        model1: The name or path of the first model, or the first tokenizer object.
        model2: The name or path of the second model, or the second tokenizer object.
        texts (list[str]): Optional. List of texts to compare. If not provided, default texts will be used.
        verbose (bool): Whether to verbosely print the tokens. Defaults to False.

    Returns:
        None: This function does not return anything.
    """
    if isinstance(model1, str):
        tokenizer1 = AutoTokenizer.from_pretrained(model1, use_fast=False)
    else:
        tokenizer1 = model1

    if isinstance(model2, str):
        tokenizer2 = AutoTokenizer.from_pretrained(model2, use_fast=False)
    else:
        tokenizer2 = model2

    if texts is None:
        texts = [
            "This is a test sentence with longer and more complicated text.",
            "Another example sentence that includes various punctuation marks, such as commas, periods, and question marks!",
            "One more sentence to compare, which contains numbers like 123 and special characters like @#$%^&*.",
            "###Human: Is this a good sentence? ###Assistant: Yes it's great!",
        ]

    same_tokens = True

    for text in texts:
        tokens1 = tokenizer1.tokenize(text)
        tokens2 = tokenizer2.tokenize(text)

        if verbose:
            print(f"Tokens from {model1}: {tokens1}")
            print(f"Tokens from {model2}: {tokens2}")

        if tokens1 != tokens2:
            same_tokens = False
            break

    same_special_tokens = check_special_tokens(tokenizer1, tokenizer2)

    if same_tokens and same_special_tokens:
        print("=============================================")
        print("Both tokenizers produce the same tokens. YAY!")
        print("=============================================")
    else:
        print("============### WARNING ###==============")
        print("Tokenizers produce DIFFERENT tokens.")
        print("============### WARNING ###==============")


if __name__ == "__main__":
    test_tokenizer_equivalence(
        "microsoft/phi-2", "Salesforce/codegen-350M-mono", verbose=True
    )

from helper.train import train_BPE


# Training
target_dataset_size = 10000

start_time = time.time()

vocab, merge_history, id_to_bytes = train_BPE(
    pretokens,
    target_dataset_size
)

end_time = time.time()

print(f"Final vocab size: {len(id_to_bytes)}")
print(f"Number of merges: {len(merge_history)}")
print(f"Training time: {end_time - start_time:.2f} seconds")

tokenizer = Tokenizer(
    vocab = id_to_bytes,
    merges = merge_history,
    special_tokens=['<|endoftext|>'],
)

print("Vocabulary size:", len(tokenizer.vocab))
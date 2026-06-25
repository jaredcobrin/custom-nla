import datasets
import random



def get_dataset_buffer():
    # load in and stream dataset
    entire_dataset = datasets.load_dataset("allenai/c4", "en", split="train", streaming=True)

    # shuffle order
    dataset = entire_dataset.shuffle(buffer_size=10000)
    dataset = iter(dataset)
    return dataset


def sample_dataset(dataset, batch_size: int, min_window_size: int, max_window_size: int):
    base_model_inputs = []
    for i in range(batch_size):
        # randomly sampling from the 10000 buffer
        text = next(dataset)["text"]

        # short text -> take entire text
        if len(text) < min_window_size:
            base_model_inputs.append(text)
        # smaller than max_window_size: take a min_window_size length
        elif len(text) < max_window_size:
            window_size = random.randint(min_window_size, len(text))
            start = random.randint(0, len(text) - window_size)
            base_model_inputs.append(text[start:start+window_size])
        # base case
        else:
            window_size = random.randint(min_window_size, max_window_size)
            start = random.randint(0, len(text) - window_size)
            base_model_inputs.append(text[start:start+window_size])
    return base_model_inputs


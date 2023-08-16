import numpy as np

def parse_vocabulary(path_text: str):
    """
    Parse vocabulary from text file.

    :param path_text: path to text file
    :return: list of words
    """
    with open(path_text, 'r') as f:
        file = f.read().replace("\n", "")

    return file.split(" ")


def create_reduced_vocabulary(path_text: str, vocab_size: int = 1000, random_state: int = 42):
    """Create a reduced vocabulary from text file.

    :param path_text: path to text file
    :param random_state: random state for reproducibility
    :return: list of words
    """
    vocabulary = parse_vocabulary(path_text)
    np.random.seed(random_state)
    return np.random.choice(vocabulary, size=vocab_size, replace=False)


def convert_list_to_vocab_text(list_of_words: list, path_text: str):
    """Convert list of words to text file.

    :param list_of_words: list of words
    :param path_text: path to text file
    :return: None
    """
    num_divisions = 1000
    num_words = len(list_of_words)
    q, r = divmod(num_words, num_divisions)
    word_sublist = []

    if r > 0:
        q += 1

    for i in range(q):
        partial_words = list_of_words[i * num_divisions: (i + 1) * num_divisions]
        print(partial_words)
        joined_partial = " ".join(partial_words)
        # print(joined_partial)
        word_sublist.append(joined_partial)
        # print(word_sublist)

    final_list = "\n".join(word_sublist)

    with open(path_text, 'w') as f:
        f.write(final_list)

    print(f"Created a new file called: {path_text}")


if __name__ == "__main__":
    # Parse vocabulary
    path_text = "./data/possible_words.txt"
    vocabulary = parse_vocabulary(path_text)
    value = False
    # for word in vocabulary:
    #     if "\n" in word:
    #         print(word)
    #         value = True
    #     else:
    #         pass
    
    # if value:
    #     print("VAMOOO")
    print(len(vocabulary))
    print(vocabulary[:10])

    reduced_vocab = create_reduced_vocabulary(path_text, vocab_size=100, random_state=42)
    convert_list_to_vocab_text(reduced_vocab, "./data/very_reduced_vocab.txt")

from datasets import load_dataset
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import DataLoader, RandomSampler, SubsetRandomSampler, TensorDataset
from torch.utils.data import  Subset
def prepare(actual_task, tokenizer, sub=False):
    task = actual_task
    dataset = load_dataset("glue", actual_task)
    global tok
    tok = tokenizer

    datataset = load_dataset('imdb')


    print("\nHow the dataset looks like:")
    print(dataset["train"][5])
    print("")

    task_to_keys = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mnli-mm": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }

    global sentence1_key, sentence2_key
    sentence1_key, sentence2_key = task_to_keys[task]

    #print('Max sentence length: ', max([len(sen) for sen in dataset["train"]["sentence"]]))
    print("Preprocess an example:")
    print(preprocess_function(dataset['train'][5]))
    print("")

    encoded_dataset = dataset.map(preprocess_function, batched=True)
    print(encoded_dataset["train"][5])

    train_inputs = pad_sequences(encoded_dataset["train"]["input_ids"], maxlen=128,
                                 dtype="long", value=0, truncating="post", padding="post")

    val_inputs = pad_sequences(encoded_dataset["validation"]["input_ids"], maxlen=128,
                                 dtype="long", value=0, truncating="post", padding="post")

    test_inputs = pad_sequences(encoded_dataset["test"]["input_ids"], maxlen=128,
                                 dtype="long", value=0, truncating="post", padding="post")

    def att_masks(input_ids):
        # Create attention masks
        attention_masks = []

        # For each sentence...
        for sent in input_ids:
            att_mask = [int(token_id > 0) for token_id in sent]
            attention_masks.append(att_mask)
        return attention_masks

    train_masks = att_masks(train_inputs)
    val_masks = att_masks(val_inputs)
    test_masks = att_masks(test_inputs)

    train_inputs = torch.tensor(train_inputs)
    val_inputs = torch.tensor(val_inputs)
    test_inputs = torch.tensor(test_inputs)

    train_labels = torch.tensor(encoded_dataset["train"]["label"])
    val_labels = torch.tensor(encoded_dataset["validation"]["label"])
    test_labels = torch.tensor(encoded_dataset["test"]["label"])

    train_masks = torch.tensor(train_masks)
    val_masks = torch.tensor(val_masks)
    test_masks = torch.tensor(test_masks)

    batch_size = 64

    # Create the DataLoader for our training set.
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    ###########################################################################################################
    if sub:
        train_data = Subset(
            train_data,
            np.random.choice(len(train_data), 10000).tolist(),
        )
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    #train_data = TensorDataset(train_inputs, train_masks, train_labels)
    # print('Dataset length: ', len(train_data))
    # subset_size = len(train_data) // 10
    # indices = torch.randperm(len(train_data)).tolist()[:subset_size]
    # train_sampler = SubsetRandomSampler(indices)
    # train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    ###########################################################################################################

    # Create the DataLoader for our validation set.
    validation_data = TensorDataset(val_inputs, val_masks, val_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    # Create the DataLoader for our test test.
    prediction_data = TensorDataset(test_inputs, test_masks, test_labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

    print("Dataset read.\n")
    return train_dataloader, validation_dataloader, prediction_dataloader



def preprocess_function(examples):
    if sentence2_key is None:
        return tok(examples[sentence1_key], truncation=True, add_special_tokens = True)
    return tok(examples[sentence1_key], examples[sentence2_key], truncation=True, add_special_tokens = True)

def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):

    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    num_samples = len(sequences)

    lengths = []
    sample_shape = ()
    flag = True

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.

    for x in sequences:
        try:
            lengths.append(len(x))
            if flag and len(x):
                sample_shape = np.asarray(x).shape[1:]
                flag = False
        except TypeError:
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))

    if maxlen is None:
        maxlen = np.max(lengths)

    is_dtype_str = np.issubdtype(dtype, np.str_) 
    if isinstance(value, str) and dtype != object and not is_dtype_str:
        raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
                         "You should set `dtype=object` for variable length strings."
                         .format(dtype, type(value)))

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x
#from glue.models_depricated import CustomRoberta
from data import prepare
import torch
import numpy as np
import torch.nn as nn 
import copy
from transformers import AutoTokenizer, TrainingArguments, Trainer
import evaluate
from transformers import get_linear_schedule_with_warmup
import torch.optim as optim
import random, time, datetime
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from models2 import LCNBertForSequenceClassification
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def count_non_zero_params(model):
    non_zero_params = sum(p.nonzero().size(0) for p in model.parameters())
    return non_zero_params

def prune_magnitude_general(model, sparsity_ratio):
    layers = model.bert.encoder.layers
    all_weights = []

    # Collect all weights from the model
    for layer in layers:
        subset = find_layers(layer)
        for name in subset:
            W = subset[name].weight.data
            all_weights.append(W.view(-1))

    # Concatenate all weights into a single tensor
    all_weights = torch.cat(all_weights)
    # Calculate the global threshold
    global_thresh = torch.sort(torch.abs(all_weights).cuda())[0][int(all_weights.numel() * sparsity_ratio)].cpu()

    # Prune weights based on the global threshold
    for layer in layers:
        subset = find_layers(layer)
        for name in subset:
            W = subset[name].weight.data
            W_metric = torch.abs(W)
            non_zero_mask = (W != 0)
            W_mask = (W_metric <= global_thresh) & non_zero_mask
            W[W_mask] = 0
    return model

def prune_movement_general(model, sparsity_ratio):
    layers = model.bert.encoder.layers
    all_weights = []
    all_gradients = []
    masks = {}

    # Collect all weights and their gradients from the model
    for layer in layers:
        subset = find_layers(layer)
        for name in subset:
            W = subset[name].weight.data
            G = subset[name].weight.grad
            all_weights.append(W.view(-1))
            all_gradients.append(G.view(-1))

    # Concatenate all weights and gradients into single tensors
    all_weights = torch.cat(all_weights)
    all_gradients = torch.cat(all_gradients)

    # Calculate the movement score
    movement_scores = all_weights * all_gradients

    # Create a mask to exclude already zeroed weights
    non_zero_mask = all_weights != 0
    non_zero_movement_scores = movement_scores[non_zero_mask]
    num_weights_to_prune = int(non_zero_movement_scores.numel() * sparsity_ratio)

    global_thresh = torch.sort(torch.abs(non_zero_movement_scores).cuda())[0][num_weights_to_prune].cpu()

    for layer in layers:
        subset = find_layers(layer)
        for name in subset:
            W = subset[name].weight.data
            G = subset[name].weight.grad
            movement_score = W * G
            non_zero_mask = W != 0
            zero_count = torch.sum(~non_zero_mask)
            #print(f"Number of zero elements: {zero_count}")
            W_mask = (torch.abs(movement_score) > global_thresh) 
            W[~W_mask] = 0
            masks[subset[name].weight] = W_mask
            #masks[name] = W_mask

    return model, masks


def apply_masks_to_optimizer(optimizer, masks):
    for group in optimizer.param_groups:
        for param in group['params']:
            #print(param)
            if param in masks:
                mask = masks[param]
                param.grad.data *= mask
                param.data *= mask
            
                if 'exp_avg' in optimizer.state[param]:
                                    optimizer.state[param]['exp_avg'] *= mask
                if 'exp_avg_sq' in optimizer.state[param]:
                    optimizer.state[param]['exp_avg_sq'] *= mask

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["cola", "mnli", "mnli-mm", "mrpc",
                                                                    "qnli", "qqp", "rte", "sst2", "stsb", "wnli"])

    args = parser.parse_args()
    return args

def run():

    args = parse_args()
    task = args.task
    #GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
    actual_task = "mnli" if task == "mnli-mm" else task
    metric = evaluate.load("glue", actual_task)

    num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2

    MODEL_DIR = '../checkpoints/checkpoint-6959/model.safetensors' 
    from transformers import AutoConfig
    config_path = "../checkpoints/checkpoint-6800/config.json"
    config = AutoConfig.from_pretrained(config_path)
    model = LCNBertForSequenceClassification(config, num_labels)

    from tokenizers import BertWordPieceTokenizer, Regex, normalizers
    tokenizer = BertWordPieceTokenizer()
    tokenizer._tokenizer.normalizer = normalizers.Sequence(
        [
            normalizers.Replace(Regex("(``|'')"), '"'),
            normalizers.NFD(),
            normalizers.Lowercase(),
            normalizers.StripAccents(),
            normalizers.Replace(Regex(" {2,}"), " "),
            normalizers.Replace(Regex(r"[^\x00-\x7F]+"), ""),
        ]
    )

    TOKENIZER_PATH = '../checkpoints/tokenizer.json'
    tokenizer = BertTokenizerFast(tokenizer_file=str(TOKENIZER_PATH))
    
    from safetensors import safe_open
    tensors = {}
    with safe_open(MODEL_DIR, framework="pt", device=0) as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k) # loads the full tensor given a key

    model_state_dict = model.state_dict()
    for key, tensor in tensors.items():
        if key.startswith('module.'): key = key[7:] 
        if key in model_state_dict:
            if model_state_dict[key].shape == tensor.shape:
                model_state_dict[key] = tensor
            else:
                print(f"Shape mismatch for {key}: model shape {model_state_dict[key].shape}, tensor shape {tensor.shape}")
        else:
            print(f"Warning: Key {key} not found in model state_dict")

    model.load_state_dict(model_state_dict)

    # keys = ['sentence', 'label', 'idx', 'input_ids', 'attention_mask']
    train_dataloader, validation_dataloader, test_dataloader = prepare(actual_task, tokenizer)

    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs!")
    #     model = torch.nn.DataParallel(model)

    model.to('cuda') 

    device = 'cuda'
    metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"
    model_name = "model"

    optimizer = optim.AdamW(model.parameters(), lr = 2e-5, eps = 1e-8)
    epochs = 6
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

        # Set the seed value all over the place to make this reproducible.
    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # Store the average loss after each epoch so we can plot them.
    loss_values_1 = []
    val_acc_1 = []
    # For each epoch...
    flagg = False
    for epoch_i in range(0, epochs):

        if epoch_i == 3: model.aa = False

        #               Training
       # ========================================

        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0

        # Put the model into training mode. Don't be mislead--the call to
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...

        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 300 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels

            # keys = ['sentence', 'label', 'idx', 'input_ids', 'attention_mask']
            
        
        
            
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because
            # accumulating the gradients is "convenient while training RNNs".
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            # This will return the loss (rather than the model output) because we
            # have provided the `labels`.
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            outputs = model(input_ids=b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask,
                        labels=b_labels)

            # The call to `model` always returns a tuple, so we need to pull the
            # loss value out of the tuple.
            loss = outputs[0]
            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.

            total_loss += loss.item()
            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if step == len(train_dataloader) - 1: #epoch_i > 0:
                if epoch_i == 0: k =0.5
                elif epoch_i == 1: k = 0.5
                elif epoch_i == 2: k =0.5
                model, masks = prune_movement_general(model, k) 
                non_zero_params = count_non_zero_params(model) - 110000000
                print(f'Non-zero parameters: {non_zero_params}')
                # print(masks)
                # print(len(masks_total))
                flagg = True
                # apply_masks_to_optimizer(optimizer, masks_total)
                # optimizer.step()
                # non_zero_params = count_non_zero_params(model)
                # print(f'aaa Non-zero parameters: {non_zero_params}')
            if flagg: apply_masks_to_optimizer(optimizer, masks)
            optimizer.step()
            if step == len(train_dataloader) - 1: #step == len(train_dataloader) - 1: #epoch_i > 0:
                non_zero_params = count_non_zero_params(model) - 110000000
                print(f'after optim Non-zero parameters: {non_zero_params}')
            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)

        # Store the loss value for plotting the learning curve.
        loss_values_1.append(avg_train_loss)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        print("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # non_zero_params = count_non_zero_params(model) - 110913794
        # print(f'Full model parameters: {non_zero_params}')
        # pruned_model = copy.deepcopy(model)
        # prune_magnitude_general(pruned_model, 0.9)
        # non_zero_params = count_non_zero_params(pruned_model) - 110913794
        # print(f'Pruned model parameters: {non_zero_params}')
        # Evaluate data for one epoch
        for batch in validation_dataloader:

            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)

            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients, saving memory and
            # speeding up validation
            with torch.no_grad():

                # Forward pass, calculate logit predictions.
                # This will return the logits rather than the loss because we have
                # not provided labels.
                # token_type_ids is the same as the "segment ids", which
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here:
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                outputs = model(input_ids=b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask)

            # Get the "logits" output by the model. The "logits" are the output
            # values prior to applying an activation function like the softmax.
            logits = outputs[1]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences.
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)

            # Accumulate the total accuracy.
            eval_accuracy += tmp_eval_accuracy

            # Track the number of batches
            nb_eval_steps += 1

        val_acc_1.append(eval_accuracy/nb_eval_steps)
        # Report the final accuracy for this validation run.
        print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
        print("  Validation took: {:}".format(format_time(time.time() - t0)))
        print("")
        print("Running Test...")
        # Test Performance
        compute_metrics(model, validation_dataloader, metric, device, task)
    print("")
    print("Training complete!")

def compute_metrics(model, test_dataloader, metric, device, task):

    predictions = []
    labels = []
    intermid_predictions = [[] for _ in range(13)]
    for batch in test_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None,
                        attention_mask=b_input_mask)

        logits = outputs[1]
        intermid = outputs[2]
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        for i in range(13):
            intermid_predictions[i].append(intermid[i].detach().cpu().numpy())
        # Store predictions and true labels
        predictions.append(logits)
        labels.append(label_ids)

    predictions = [item for sublist in predictions for item in sublist]

    labels = [item for sublist in labels for item in sublist]
    for i in range(len(labels)):
        if labels[i] == -1: labels[i] = 0

    for i,inter in enumerate(intermid_predictions):
        intermid_predictions[i] = [item for sublist in inter for item in sublist]

    if task != "stsb":
        predictions = np.argmax(predictions, axis=1)
        intermid_preds = [np.argmax(p, axis=1) for p in intermid_predictions]
    else:
        predictions = logits[:, 0]
        intermid_preds = [p[:, 0] for p in intermid_predictions]

    m1 = metric.compute(predictions=predictions, references=labels)
    m2 = [metric.compute(predictions=p, references=labels) for p in intermid_preds]
    print("")
    print("Metric: ", m1)
    print("Intermid Metrics: ", m2)
    print("")

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))



if __name__ == "__main__":
    run()

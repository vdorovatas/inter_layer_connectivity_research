import numpy as np
from evaluate import load
import torch

PAD_TOKEN, PAD_INDEX = '[PAD]', -100

import numpy as np

def mlm_cross_entropy_loss(predictions, targets):
    # Convert to numpy arrays if they are not already
    predictions = np.array(predictions)
    targets = np.array(targets)
    exp_preds = np.exp(predictions - np.max(predictions, axis=-1, keepdims=True))  # For numerical stability
    softmax_preds = exp_preds / np.sum(exp_preds, axis=-1, keepdims=True)
    relevant_indexes = (targets != PAD_INDEX)
    relevant_predictions = softmax_preds[relevant_indexes]
    relevant_targets = targets[relevant_indexes]
    num_classes = predictions.shape[-1]  # Assuming last dimension of predictions is the number of classes
    one_hot_targets = np.eye(num_classes)[relevant_targets]
    epsilon = 1e-8
    log_preds = np.log(np.clip(relevant_predictions, epsilon, 1.0 - epsilon))
    loss_per_example = -np.sum(one_hot_targets * log_preds, axis=-1)
    loss = np.mean(loss_per_example)

    return loss


def mlm_accuracy(predictions, targets):
    predictions = np.array(predictions)
    targets = np.array(targets)
    mlm_predictions = np.argmax(predictions, axis=-1)
#     print(mlm_predictions[1])
#     print(targets[1][1])
    relevant_indexes = (targets != PAD_INDEX)
    relevant_predictions = mlm_predictions[relevant_indexes]
    relevant_targets = targets[relevant_indexes]
#     print(relevant_predictions)
#     print(relevant_targets)
#     print('')
    corrects = (relevant_predictions == relevant_targets).astype(np.float32)
    accuracy = corrects.sum() / corrects.size

    return accuracy




def nsp_accuracy(predictions, targets):
        mlm_predictions, nsp_predictions = predictions
        mlm_targets, is_nexts = targets

        corrects = np.equal(nsp_predictions, is_nexts)
        return corrects.mean()

def my_metrics(eval_pred):
       predictions, targets = eval_pred[0], eval_pred[1]
       mlm_acc_intermid = [round(float(mlm_cross_entropy_loss(p, targets)),4) for p in predictions]
       #mlm_acc = mlm_accuracy(predictions, targets)
       print('\nMLM ACC ALL LAYERS: ')
       print(mlm_acc_intermid)
       print('')
       return mlm_acc_intermid[-1]

def classification_accuracy(predictions, targets):
        corrects = np.equal(predictions, targets)
        return corrects.mean()
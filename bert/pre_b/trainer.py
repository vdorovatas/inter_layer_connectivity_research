from transformers import Trainer
import torch
from typing import Dict, Union, Any, List, Tuple
from torch import nn
from typing import Optional
from collections.abc import Mapping
import time 
#from .logging import *
from torch.utils.data import DataLoader, Dataset, IterableDataset
import numpy as np
from typing import NamedTuple
from transformers.utils import is_torch_available

# logger = logging.get_logger(__name__)

class IterableDatasetShard(IterableDataset):
    """
    Wraps a PyTorch `IterableDataset` to generate samples for one of the processes only. Instances of this class will
    always yield a number of samples that is a round multiple of the actual batch size (which is `batch_size x
    num_processes`). Depending on the value of the `drop_last` attribute, it will either stop the iteration at the
    first batch that would be too small or loop with indices from the beginning.

    On two processes with an iterable dataset yielding of `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]` with a batch size of
    2:

    - the shard on process 0 will yield `[0, 1, 4, 5, 8, 9]` so will see batches `[0, 1]`, `[4, 5]`, `[8, 9]`
    - the shard on process 1 will yield `[2, 3, 6, 7, 10, 11]` so will see batches `[2, 3]`, `[6, 7]`, `[10, 11]`

    <Tip warning={true}>

        If your IterableDataset implements some randomization that needs to be applied the same way on all processes
        (for instance, a shuffling), you should use a `torch.Generator` in a `generator` attribute of the `dataset` to
        generate your random numbers and call the [`~trainer_pt_utils.IterableDatasetShard.set_epoch`] method of this
        object. It will set the seed of this `generator` to `seed + epoch` on all processes before starting the
        iteration. Alternatively, you can also implement a `set_epoch()` method in your iterable dataset to deal with
        this.

    </Tip>

    Args:
        dataset (`torch.utils.data.IterableDataset`):
            The batch sampler to split in several shards.
        batch_size (`int`, *optional*, defaults to 1):
            The size of the batches per shard.
        drop_last (`bool`, *optional*, defaults to `False`):
            Whether or not to drop the last incomplete batch or complete the last batches by using the samples from the
            beginning.
        num_processes (`int`, *optional*, defaults to 1):
            The number of processes running concurrently.
        process_index (`int`, *optional*, defaults to 0):
            The index of the current process.
        seed (`int`, *optional*, defaults to 0):
            A random seed that will be used for the random number generation in
            [`~trainer_pt_utils.IterableDatasetShard.set_epoch`].
    """

    def __init__(
        self,
        dataset: IterableDataset,
        batch_size: int = 1,
        drop_last: bool = False,
        num_processes: int = 1,
        process_index: int = 0,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_processes = num_processes
        self.process_index = process_index
        self.seed = seed
        self.epoch = 0
        self.num_examples = 0

    def set_epoch(self, epoch):
        self.epoch = epoch
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

    def __iter__(self):
        self.num_examples = 0
        if (
            not hasattr(self.dataset, "set_epoch")
            and hasattr(self.dataset, "generator")
            and isinstance(self.dataset.generator, torch.Generator)
        ):
            self.dataset.generator.manual_seed(self.seed + self.epoch)
        real_batch_size = self.batch_size * self.num_processes
        process_slice = range(self.process_index * self.batch_size, (self.process_index + 1) * self.batch_size)

        first_batch = None
        current_batch = []
        for element in self.dataset:
            self.num_examples += 1
            current_batch.append(element)
            # Wait to have a full batch before yielding elements.
            if len(current_batch) == real_batch_size:
                for i in process_slice:
                    yield current_batch[i]
                if first_batch is None:
                    first_batch = current_batch.copy()
                current_batch = []

        # Finished if drop_last is True, otherwise complete the last batch with elements from the beginning.
        if not self.drop_last and len(current_batch) > 0:
            if first_batch is None:
                first_batch = current_batch.copy()
            while len(current_batch) < real_batch_size:
                current_batch += first_batch
            for i in process_slice:
                yield current_batch[i]

    def __len__(self):
        # Will raise an error if the underlying dataset is not sized.
        if self.drop_last:
            return (len(self.dataset) // (self.batch_size * self.num_processes)) * self.batch_size
        else:
            return math.ceil(len(self.dataset) / (self.batch_size * self.num_processes)) * self.batch_size


def atleast_1d(tensor_or_array: Union[torch.Tensor, np.ndarray]):
    if isinstance(tensor_or_array, torch.Tensor):
        if hasattr(torch, "atleast_1d"):
            tensor_or_array = torch.atleast_1d(tensor_or_array)
        elif tensor_or_array.ndim < 1:
            tensor_or_array = tensor_or_array[None]
    else:
        tensor_or_array = np.atleast_1d(tensor_or_array)
    return tensor_or_array

def torch_pad_and_concatenate(tensor1, tensor2, padding_index=-100):
    """Concatenates `tensor1` and `tensor2` on first axis, applying padding on the second if necessary."""
    tensor1 = atleast_1d(tensor1)
    tensor2 = atleast_1d(tensor2)

    if len(tensor1.shape) == 1 or tensor1.shape[1] == tensor2.shape[1]:
        return torch.cat((tensor1, tensor2), dim=0)

    # Let's figure out the new shape
    new_shape = (tensor1.shape[0] + tensor2.shape[0], max(tensor1.shape[1], tensor2.shape[1])) + tensor1.shape[2:]

    # Now let's fill the result tensor
    result = tensor1.new_full(new_shape, padding_index)
    result[: tensor1.shape[0], : tensor1.shape[1]] = tensor1
    result[tensor1.shape[0] :, : tensor2.shape[1]] = tensor2
    return result

class EvalPrediction:
    """
    Evaluation output (always contains labels), to be used to compute metrics.

    Parameters:
        predictions (`np.ndarray`): Predictions of the model.
        label_ids (`np.ndarray`): Targets to be matched.
        inputs (`np.ndarray`, *optional*): Input data passed to the model.
        losses (`np.ndarray`, *optional*): Loss values computed during evaluation.
    """

    def __init__(
        self,
        predictions: Union[np.ndarray, Tuple[np.ndarray]],
        label_ids: Union[np.ndarray, Tuple[np.ndarray]],
        inputs: Optional[Union[np.ndarray, Tuple[np.ndarray]]] = None,
        losses: Optional[Union[np.ndarray, Tuple[np.ndarray]]] = None,
    ):
        self.predictions = predictions
        self.label_ids = label_ids
        self.inputs = inputs
        self.losses = losses
        self.elements = (self.predictions, self.label_ids)
        if self.inputs is not None:
            self.elements += (self.inputs,)
        if self.losses is not None:
            self.elements += (self.losses,)

    def __iter__(self):
        return iter(self.elements)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.elements):
            raise IndexError("tuple index out of range")
        return self.elements[idx]


def numpy_pad_and_concatenate(array1, array2, padding_index=-100):
    """Concatenates `array1` and `array2` on first axis, applying padding on the second if necessary."""
    array1 = atleast_1d(array1)
    array2 = atleast_1d(array2)

    if len(array1.shape) == 1 or array1.shape[1] == array2.shape[1]:
        return np.concatenate((array1, array2), axis=0)

    # Let's figure out the new shape
    new_shape = (array1.shape[0] + array2.shape[0], max(array1.shape[1], array2.shape[1])) + array1.shape[2:]

    # Now let's fill the result tensor
    result = np.full_like(array1, padding_index, shape=new_shape)
    result[: array1.shape[0], : array1.shape[1]] = array1
    result[array1.shape[0] :, : array2.shape[1]] = array2
    return result



def nested_concat(tensors, new_tensors, padding_index=-100):
    """
    Concat the `new_tensors` to `tensors` on the first dim and pad them on the second if needed. Works for tensors or
    nested list/tuples/dict of tensors.
    """
    if not (isinstance(tensors, torch.Tensor) and isinstance(new_tensors, torch.Tensor)):
        assert (
            type(tensors) is type(new_tensors)
        ), f"Expected `tensors` and `new_tensors` to have the same type but found {type(tensors)} and {type(new_tensors)}."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_concat(t, n, padding_index=padding_index) for t, n in zip(tensors, new_tensors))
    elif isinstance(tensors, torch.Tensor):
        return torch_pad_and_concatenate(tensors, new_tensors, padding_index=padding_index)
    elif isinstance(tensors, Mapping):
        return type(tensors)(
            {k: nested_concat(t, new_tensors[k], padding_index=padding_index) for k, t in tensors.items()}
        )
    elif isinstance(tensors, np.ndarray):
        return numpy_pad_and_concatenate(tensors, new_tensors, padding_index=padding_index)
    else:
        raise TypeError(f"Unsupported type for concatenation: got {type(tensors)}")

def nested_numpify(tensors):
    "Numpify `tensors` (even if it's a nested list/tuple/dict of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_numpify(t) for t in tensors)
    if isinstance(tensors, Mapping):
        return type(tensors)({k: nested_numpify(t) for k, t in tensors.items()})

    t = tensors.cpu()
    if t.dtype == torch.bfloat16:
        # As of Numpy 1.21.4, NumPy does not support bfloat16 (see
        # https://github.com/numpy/numpy/blob/a47ecdea856986cd60eabbd53265c2ca5916ad5d/doc/source/user/basics.types.rst ).
        # Until Numpy adds bfloat16, we must convert float32.
        t = t.to(torch.float32)
    return t.numpy()

class EvalLoopContainer:
    """
    Container to store intermediate results of evaluation loop

    Args:
        do_nested_concat (`bool`, *optional*, defaults to `True`):
            If set to `True`, each iteration will recursively concatenate a new object containing tensors to
            the existing stored tensors, provided that the structure of the existing object and the new one
            are identical. If set to `False`, all newly added tensors will be stored in a list.
        padding_index (`int`, *optional*, defaults to -100):
            Value used to pad tensors of different shapes when `do_nested_concat=True`.
    """

    def __init__(self, do_nested_concat: bool = True, padding_index: int = -100):
        self.do_nested_concat = do_nested_concat
        self.padding_index = padding_index
        self.tensors = None
        self.arrays = None

    def add(self, tensors) -> None:
        """Add tensors to the stored objects. If `do_nested_concat=True`, the tensors will be concatenated recursively."""
        if self.tensors is None:
            self.tensors = tensors if self.do_nested_concat else [tensors]
        elif self.do_nested_concat:
            self.tensors = nested_concat(self.tensors, tensors, padding_index=self.padding_index)
        else:
            self.tensors.append(tensors)

    def to_cpu_and_numpy(self) -> None:
        """Move tensors in stored objects to CPU and convert them to numpy arrays."""

        # Check if we have something to add, if not just return
        if self.tensors is None:
            return

        new_arrays = nested_numpify(self.tensors)
        if self.arrays is None:
            self.arrays = new_arrays
        elif self.do_nested_concat:
            self.arrays = nested_concat(self.arrays, new_arrays, padding_index=self.padding_index)
        else:
            self.arrays.extend(new_arrays)

        # reset device tensors after adding to cpu
        self.tensors = None

    def get_arrays(self):
        """Returns the numpified and moved to CPU stored objects."""
        self.to_cpu_and_numpy()
        return self.arrays


def denumpify_detensorize(metrics):
    """
    Recursively calls `.item()` on the element of the dictionary passed
    """
    if isinstance(metrics, (list, tuple)):
        return type(metrics)(denumpify_detensorize(m) for m in metrics)
    elif isinstance(metrics, dict):
        return type(metrics)({k: denumpify_detensorize(v) for k, v in metrics.items()})
    elif isinstance(metrics, np.generic):
        return metrics.item()
    elif is_torch_available() and isinstance(metrics, torch.Tensor) and metrics.numel() == 1:
        return metrics.item()
    return metrics


def has_length(dataset):
    """
    Checks if the dataset implements __len__() and it doesn't raise an error
    """
    try:
        return len(dataset) is not None
    except TypeError:
        # TypeError: len() of unsized object
        return False
    
def find_batch_size(tensors):
    """
    Find the first dimension of a tensor in a nested list/tuple/dict of tensors.
    """
    if isinstance(tensors, (list, tuple)):
        for t in tensors:
            result = find_batch_size(t)
            if result is not None:
                return result
    elif isinstance(tensors, Mapping):
        for key, value in tensors.items():
            result = find_batch_size(value)
            if result is not None:
                return result
    elif isinstance(tensors, torch.Tensor):
        return tensors.shape[0] if len(tensors.shape) >= 1 else None
    elif isinstance(tensors, np.ndarray):
        return tensors.shape[0] if len(tensors.shape) >= 1 else None

    
class EvalLoopOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]
    num_samples: Optional[int]

class CustomTrainer(Trainer):
    def __init__(self, lcn, *args, **kwargs):
        self.lcn = lcn
        super().__init__(*args, **kwargs)
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if False: #self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            start_time = time.time()
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled or self.is_fsdp_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )
            self.model_preparation_time = round(time.time() - start_time, 4)

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        # logger.info(f"\n***** Running {description} *****")
        # if has_length(dataloader):
        #     logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        # else:
        #     logger.info("  Num examples: Unknown")
        # logger.info(f"  Batch size = {batch_size}")

        model.eval()
        if hasattr(self.optimizer, "eval") and callable(self.optimizer.eval):
            self.optimizer.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        all_losses = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_preds = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_labels = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_inputs = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_logits_intermid_list = [EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100) for _ in range(13)]

        metrics = None
        eval_set_kwargs = {}

        # Will be useful when we have an iterable dataset so don't know its length.
        observed_num_examples = 0

        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            losses, logits, labels, logits_intermid = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = (
                self._prepare_input(inputs[main_input_name]) if "inputs" in args.include_for_metrics else None
            )

            # if is_torch_xla_available():
            #     xm.mark_step()

            # Update containers
            if losses is not None:
                losses = self.gather_function((losses.repeat(batch_size)))
                all_losses.add(losses)
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.gather_function((inputs_decode))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_inputs.add(inputs_decode)
            if labels is not None:
                # Pad labels here, preparing for preprocess_logits_for_metrics in next logits block.
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.gather_function((logits))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_preds.add(logits)
            if labels is not None:
                labels = self.gather_function((labels))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_labels.add(labels)
            ######
            if logits_intermid is not None:
                for i,logits in enumerate(logits_intermid):
                    logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                    if self.preprocess_logits_for_metrics is not None:
                        logits = self.preprocess_logits_for_metrics(logits, labels)
                    logits = self.gather_function((logits))
                    if not self.args.batch_eval_metrics or description == "Prediction":
                        all_logits_intermid_list[i].add(logits)
                        all_logits_intermid_list[i].to_cpu_and_numpy()
            
            #####
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            if self.args.batch_eval_metrics:
                print('batch_eval_metrics')
                if self.compute_metrics is not None and logits is not None and labels is not None:
                    is_last_step = self.accelerator.gradient_state.end_of_dataloader
                    batch_kwargs = {}
                    batch_kwargs["losses"] = losses if "loss" in args.include_for_metrics else None
                    batch_kwargs["inputs"] = inputs if "inputs" in args.include_for_metrics else None
                    metrics = self.compute_metrics(
                        EvalPrediction(predictions=logits, label_ids=labels, **batch_kwargs),
                        compute_result=is_last_step,
                    )

                del losses, logits, labels, inputs
                torch.cuda.empty_cache()

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            elif args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                all_losses.to_cpu_and_numpy()
                all_preds.to_cpu_and_numpy()
                all_labels.to_cpu_and_numpy()
                all_inputs.to_cpu_and_numpy()
                # for l in all_logits_intermid_list: 
                #     l.to_cpu_and_numpy()

                del losses, logits, labels, inputs
                torch.cuda.empty_cache()

        # After all calls to `.gather_function`, reset to `gather_for_metrics`:
        self.gather_function = self.accelerator.gather_for_metrics
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        all_losses = all_losses.get_arrays()
        all_preds = all_preds.get_arrays()
        all_labels = all_labels.get_arrays()
        all_inputs = all_inputs.get_arrays()
        all_logits_intermid = []
        for l in all_logits_intermid_list: 
            all_logits_intermid.append(l.get_arrays())

        del all_logits_intermid_list

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Metrics!
        if (
            self.compute_metrics is not None
            and all_preds is not None
            and all_labels is not None
            and not self.args.batch_eval_metrics
        ):
            print('in metrics')
            eval_set_kwargs["losses"] = all_losses if "loss" in args.include_for_metrics else None
            eval_set_kwargs["inputs"] = all_inputs if "inputs" in args.include_for_metrics else None
            metrics = self.compute_metrics(
                EvalPrediction(predictions=all_logits_intermid, label_ids=all_labels, **eval_set_kwargs)
            )
        elif metrics is None:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        print('LOSS: ', all_losses.mean().item())

        if isinstance(all_losses, list) and all_losses:
            metrics[f"{metric_key_prefix}_loss"] = np.concatenate(all_losses).mean().item()
        elif isinstance(all_losses, np.ndarray):
            metrics = {} 
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time
        if hasattr(self, "model_preparation_time"):
            metrics[f"{metric_key_prefix}_model_preparation_time"] = self.model_preparation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)
        #return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        self.label_names = ['labels']
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        #print('HAS_LABELS: ', has_labels)
        with torch.no_grad():
            if has_labels or loss_without_labels:
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()

                ##########
                if self.lcn:
                    hidden_states = outputs.hidden_states
                    current = hidden_states[0]
                    #intermid = [model.module.cls(current)]
                    if hasattr(model, 'module'): intermid = [model.module.cls(current)]
                    else: intermid = [model.cls(current)]
                    for h in hidden_states[1:]:
                        current = current + h
                        #intermid.append(model.module.cls(current))
                        if hasattr(model, 'module'): intermid.append(model.module.cls(current))
                        else: intermid.append(model.cls(current))
                else:
                    hidden_states = outputs.hidden_states
                    #intermid = [model.module.cls(h) for h in hidden_states]
                    if hasattr(model, 'module'): intermid = [model.module.cls(h) for h in hidden_states]
                    else: intermid = [model.cls(h) for h in hidden_states]
                #########
                logits = outputs.logits
                del hidden_states

            else:
                print('here')
                loss = None
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                else:
                    logits = outputs
                # TODO: this needs to be fixed and made cleaner later.
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]
        
        intermid_logits = []
        for il in intermid:
            il = nested_detach(il)
            if len(il) == 1:
                il = il[0]
            intermid_logits.append(il)

        # print(labels)
        # print('\n\n\n')
        # print(logits)
            
        return (loss, logits, labels, intermid_logits)

    
def nested_detach(tensors):
    "Detach `tensors` (even if it's a nested list/tuple/dict of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_detach(t) for t in tensors)
    elif isinstance(tensors, Mapping):
        return type(tensors)({k: nested_detach(t) for k, t in tensors.items()})
    return tensors.detach() if isinstance(tensors, torch.Tensor) else tensors

"""
scan_data.py — SCAN dataset loader for decoder-only LM experiments.

Format per example (single sequence):
    IN: <command> OUT: <action sequence> <eos>

Loss is computed only on the OUT side (command tokens are masked with -1).
Download splits from: https://github.com/brendenlake/SCAN
"""

import os
import random
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Tokenizer — character-level over SCAN's small vocabulary
# ---------------------------------------------------------------------------

class SCANTokenizer:
    """
    Simple word-level tokenizer over the SCAN vocabulary.
    Keeps vocab small (< 50 tokens) so embedding table is tiny and
    the model is not spending capacity on tokenization.
    """

    PAD   = "<pad>"
    BOS   = "<bos>"
    EOS   = "<eos>"
    SPECIAL = [PAD, BOS, EOS]

    def __init__(self):
        # Complete SCAN vocabulary (input + output sides)
        base_vocab = [
            # input side
            "IN:", "OUT:",
            "jump", "run", "walk", "look", "turn",
            "left", "right", "around", "opposite", "twice", "thrice",
            "and", "after",
            # output side — all I_ actions SCAN can produce
            "I_JUMP", "I_RUN", "I_WALK", "I_LOOK",
            "I_TURN_LEFT", "I_TURN_RIGHT",
        ]
        vocab = self.SPECIAL + base_vocab
        self.token2id: Dict[str, int] = {t: i for i, t in enumerate(vocab)}
        self.id2token: Dict[int, str] = {i: t for t, i in self.token2id.items()}
        self.pad_id = self.token2id[self.PAD]
        self.bos_id = self.token2id[self.BOS]
        self.eos_id = self.token2id[self.EOS]

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)

    def encode(self, text: str) -> List[int]:
        return [self.token2id[t] for t in text.strip().split()]

    def decode(self, ids: List[int]) -> str:
        return " ".join(self.id2token.get(i, "?") for i in ids
                        if i not in (self.pad_id, self.bos_id))


# ---------------------------------------------------------------------------
# Parsing raw SCAN files
# ---------------------------------------------------------------------------

def parse_scan_file(path: str) -> List[Tuple[str, str]]:
    """Return list of (command, action_sequence) string pairs."""
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # format: "IN: jump OUT: JUMP"
            assert line.startswith("IN:") and "OUT:" in line, f"Bad line: {line}"
            _, rest = line.split("IN:", 1)
            cmd, act = rest.split("OUT:", 1)
            pairs.append((cmd.strip(), act.strip()))
    return pairs


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SCANDataset(Dataset):
    """
    Each item is a packed sequence fed to a causal LM:

        input : [BOS] IN: <cmd> OUT: <act_1> ... <act_n>
        labels:  -1    -1  -1    -1   <act_1> ... <act_n> [EOS]

    labels[t] = token the model should predict at position t = seq[t+1].
    Positions up to and including "OUT:" are masked with -1 (no loss).
    Loss is computed only on the action tokens + EOS.
    """

    def __init__(self, pairs: List[Tuple[str, str]], tokenizer: SCANTokenizer,
                 max_len: int = 128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples = []

        for cmd, act in pairs:
            cmd_ids = tokenizer.encode("IN: " + cmd)   # includes "IN:" token
            act_ids = tokenizer.encode("OUT: " + act)  # includes "OUT:" token

            # full input sequence (what the model sees):
            # [BOS] IN: w1 w2 ... OUT: a1 a2 ... an
            seq = [tokenizer.bos_id] + cmd_ids + act_ids

            # target sequence (shifted by 1):
            # predict next token at every position
            # mask everything up to and including OUT: (the separator)
            # OUT: sits at position: 1 + len(cmd_ids)  (0-indexed in seq)
            # so the first action token a1 is at position 1 + len(cmd_ids) + 1
            # labels[t] = seq[t+1], but -1 for all t where seq[t+1] <= OUT: position

            full_target = seq[1:] + [tokenizer.eos_id]  # shifted right by 1

            # number of input positions to mask: BOS + cmd_ids + "OUT:" token
            # = model sees those, but we don't compute loss on predicting them
            n_masked = len(cmd_ids) + 1  # +1 for "OUT:" which is act_ids[0]
            # in full_target terms: mask positions 0 .. n_masked-1
            labels = [-1] * n_masked + full_target[n_masked:]

            # 🔍 DEBUG (only for first few examples)
            if False:
                print("\n--- DEBUG EXAMPLE ---")
                print("SEQ:    ", tokenizer.decode(seq))
                print("TARGET: ", [
                    tokenizer.id2token[t] if t != -1 else "_"
                    for t in labels
                ])

            assert len(seq) == len(labels), \
                f"seq len {len(seq)} != labels len {len(labels)}"

            if len(seq) > max_len:
                continue

            self.examples.append((seq, labels))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        seq, labels = self.examples[idx]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def collate_fn(batch, pad_id: int):
    seqs, labels = zip(*batch)
    max_len = max(s.size(0) for s in seqs)
    padded_seqs   = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    padded_labels = torch.full((len(seqs), max_len), -1,     dtype=torch.long)
    for i, (s, l) in enumerate(zip(seqs, labels)):
        padded_seqs[i,   :s.size(0)] = s
        padded_labels[i, :l.size(0)] = l
    return padded_seqs, padded_labels


# ---------------------------------------------------------------------------
# Split loader
# ---------------------------------------------------------------------------

SPLIT_FILES = {
    # (subfolder, train_file, test_file)
    "simple":             ("simple_split",   "tasks_train_simple.txt",            "tasks_test_simple.txt"),
    "length":             ("length_split",   "tasks_train_length.txt",            "tasks_test_length.txt"),
    "add_prim_jump":      ("add_prim_split", "tasks_train_addprim_jump.txt",      "tasks_test_addprim_jump.txt"),
    "add_prim_turn_left": ("add_prim_split", "tasks_train_addprim_turn_left.txt", "tasks_test_addprim_turn_left.txt"),
    "add_prim_jump_num2_rep1": ("add_prim_split/with_additional_examples", "tasks_train_addprim_complex_jump_num2_rep1.txt", "tasks_test_addprim_complex_jump_num2_rep1.txt")  ,                                                                                                                                             "add_prim_jump_num2_rep2": ("add_prim_split/with_additional_examples", "tasks_train_addprim_complex_jump_num2_rep2.txt", "tasks_test_addprim_complex_jump_num2_rep2.txt"),
    "add_prim_jump_num4_rep1": ("add_prim_split/with_additional_examples", "tasks_train_addprim_complex_jump_num4_rep1.txt", "tasks_test_addprim_complex_jump_num4_rep1.txt")  ,                                                                                                                                             "add_prim_jump_num4_rep2": ("add_prim_split/with_additional_examples", "tasks_train_addprim_complex_jump_num4_rep2.txt", "tasks_test_addprim_complex_jump_num4_rep2.txt"),
    "add_prim_jump_num8_rep1": ("add_prim_split/with_additional_examples", "tasks_train_addprim_complex_jump_num8_rep1.txt", "tasks_test_addprim_complex_jump_num8_rep1.txt"),
    "add_prim_jump_num8_rep2": ("add_prim_split/with_additional_examples", "tasks_train_addprim_complex_jump_num8_rep2.txt", "tasks_test_addprim_complex_jump_num8_rep2.txt"),
    "add_prim_jump_num16_rep1": ("add_prim_split/with_additional_examples", "tasks_train_addprim_complex_jump_num16_rep1.txt", "tasks_test_addprim_complex_jump_num16_rep1.txt"),
    "add_prim_jump_num16_rep2": ("add_prim_split/with_additional_examples", "tasks_train_addprim_complex_jump_num16_rep2.txt", "tasks_test_addprim_complex_jump_num16_rep2.txt"),
    "add_prim_jump_num32_rep1": ("add_prim_split/with_additional_examples", "tasks_train_addprim_complex_jump_num32_rep1.txt", "tasks_test_addprim_complex_jump_num32_rep1.txt"),
    "add_prim_jump_num32_rep2": ("add_prim_split/with_additional_examples", "tasks_train_addprim_complex_jump_num32_rep2.txt", "tasks_test_addprim_complex_jump_num32_rep2.txt"),
    "simple_p1": ("simple_split/size_variations",  "tasks_train_simple_p1.txt", "tasks_test_simple_p1.txt"),
    "simple_p2": ("simple_split/size_variations",  "tasks_train_simple_p2.txt", "tasks_test_simple_p2.txt"),
    "simple_p4": ("simple_split/size_variations",  "tasks_train_simple_p4.txt", "tasks_test_simple_p4.txt"),
    "simple_p8": ("simple_split/size_variations",  "tasks_train_simple_p8.txt", "tasks_test_simple_p8.txt"),
    "simple_p16": ("simple_split/size_variations",  "tasks_train_simple_p16.txt", "tasks_test_simple_p16.txt"),
    "simple_p32": ("simple_split/size_variations",  "tasks_train_simple_p32.txt", "tasks_test_simple_p32.txt"),
}


def get_dataloaders(scan_dir: str, split: str, tokenizer: SCANTokenizer,
                    batch_size: int, max_len: int = 128,
                    num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
    """
    Returns (train_loader, test_loader) for a given split name.
    scan_dir: path to the cloned SCAN repo root (contains the .txt files).
    """
    subfolder, train_file, test_file = SPLIT_FILES[split]

    def find(fname):
        path = os.path.join(scan_dir, subfolder, fname)
        if os.path.exists(path):
            return path
        raise FileNotFoundError(f"Could not find {fname} under {scan_dir}/{subfolder}")

    train_pairs = parse_scan_file(find(train_file))
    test_pairs  = parse_scan_file(find(test_file))

    train_ds = SCANDataset(train_pairs, tokenizer, max_len=max_len)
    test_ds  = SCANDataset(test_pairs,  tokenizer, max_len=max_len)

    _collate = lambda b: collate_fn(b, tokenizer.pad_id)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=_collate, num_workers=num_workers,
                              drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=2*batch_size, shuffle=False,
                              collate_fn=_collate, num_workers=num_workers)

    print(f"[{split}] train={len(train_ds)}  test={len(test_ds)}")
    return train_loader, test_loader

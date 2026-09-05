import re
from collections import defaultdict

log_file = "outputs/lsn_08_add_prim_jump_with_additional_examples.log"  # change to your file

# pattern for split
split_pattern = re.compile(r"Split:\s+(jump_num\d+_rep\d+)")

# pattern for EM values (layer lines + FINAL)
em_pattern = re.compile(r"EM=([\d\.]+)%")

best_em = defaultdict(float)
current_split = None

with open(log_file, "r") as f:
    for line in f:
        # detect split
        split_match = split_pattern.search(line)
        if split_match:
            current_split = split_match.group(1)
            continue

        if current_split is None:
            continue

        # find EM values in line
        em_match = em_pattern.search(line)
        if em_match:
            em = float(em_match.group(1))
            if em > best_em[current_split]:
                best_em[current_split] = em

# print results
print("Best EM per split:\n")
for split in sorted(best_em.keys()):
    print(f"{split}: {best_em[split]:.2f}%")

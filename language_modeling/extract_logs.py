import re
import sys

def parse_log(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Split into blocks by "## Intermid [val] PPL ##"
    # Each block should contain one train iteration line before it and one PPL list after it

    steps = []

    # Find all train iteration lines
    train_pattern = re.compile(
        r'global_step=(\d+).*?\[train\] loss=([\d.]+).*?\[val\] loss=([\d.]+)',
        re.DOTALL
    )

    # Find all Intermid PPL lists
    ppl_pattern = re.compile(
        r'## Intermid \[val\] PPL ##\s*\n\[(.*?)\]',
        re.DOTALL
    )

    train_matches = list(train_pattern.finditer(content))
    ppl_matches = list(ppl_pattern.finditer(content))

    if len(train_matches) != len(ppl_matches):
        print(f"Warning: found {len(train_matches)} train entries but {len(ppl_matches)} PPL lists. Pairing by order.")

    results = []
    for i, (tm, pm) in enumerate(zip(train_matches, ppl_matches)):
        step = int(tm.group(1))
        train_loss = float(tm.group(2))
        val_loss = float(tm.group(3))

        # Parse PPL list — strip quotes and whitespace
        raw_items = pm.group(1).split(',')
        ppl_list = [float(x.strip().strip("'\"")) for x in raw_items if x.strip()]

        results.append({
            'step': step,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'intermid_ppl': ppl_list[-1],
        })

    return results


def main():
    filepath = sys.argv[1] if len(sys.argv) > 1 else 'train.log'
    results = parse_log(filepath)

    steps       = [r['step']        for r in results]
    train_losses = [r['train_loss'] for r in results]
    val_losses   = [r['val_loss']   for r in results]
    intermid_ppls = [r['intermid_ppl'] for r in results]

    print("Steps:")
    print(steps)
    print()
    print("Train losses:")
    print(train_losses)
    print()
    print("Val losses:")
    print(val_losses)
    print()
    print("Intermediate val PPL - final layer (one value per step):")
    print(intermid_ppls)


if __name__ == '__main__':
    main()

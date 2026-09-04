train_losses = []

with open("new_outputs/acn_300ep.out", "r") as f:
    for line in f:
        if "Train loss:" in line:
            # split and grab the number
            loss_value = float(line.strip().split()[-1])
            train_losses.append(loss_value)

print(train_losses)

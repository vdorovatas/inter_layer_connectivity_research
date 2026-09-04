final_intermediate_accs = []

with open("outputs/acn_700epochs_lr4e-4_w80k_resume.out", "r") as f:
    for line in f:
        if line.startswith("Intermid Top-1 Accuracies:"):
            # read the next line (the list itself)
            acc_line = next(f).strip()
            # convert string to actual Python list
            acc_list = eval(acc_line)  
            # take the last element
            final_intermediate_accs.append(acc_list[-1])

print(final_intermediate_accs)


"""The CarliniWagnerL0 attack."""

import torch
from CW_L2 import carlini_wagner_l2

def carlini_wagner_l0(
    model_fn,
    x,
    n_classes,
    y=None,
    targeted=False,
    lr=5e-3,
    confidence=0,
    clip_min=0,
    clip_max=1,
    initial_const=1e-2,
    binary_search_steps=5,
    max_iterations=1000,
):

    # Define loss functions and optimizer
    def f(x):
        logits = model_fn(x)
        y_adv = torch.argmax(model(adversarial_images), 1) # use labels of adversarials as y
        y_onehot = torch.nn.functional.one_hot(y_adv, n_classes).to(torch.float)
        real = torch.sum(y_onehot * logits, 1)
        other, _ = torch.max((1 - y_onehot) * logits - y_onehot * 1e4, 1)
        if targeted:
            return torch.max((other - real),torch.tensor(0.0))
        else:
            return torch.max((real - other) + confidence, torch.tensor(0.0))

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        x = x.to(device)
        pred = model_fn(x)
        y = torch.argmax(pred, 1)

    no_change = torch.ones_like(x)

    step = 0
    adversarial_images = x.clone()

    done = torch.zeros_like(y)

    while torch.sum(no_change) != 0:

        adversarial_images = carlini_wagner_l2(model, x, adversarial_images, n_classes, no_change, y=y ,targeted=targeted)
        # if l2 doesnt find something it should return the last adversarial since we hand it to l2 for the warm start
        predictions = torch.argmax(model(adversarial_images),1)
        for i in range(len(predictions)):
            if (done[i] == 0) and (predictions[i] == y[i]):
                print("No new adversarial for image "+str(i))
                done[i] = 1

        if torch.sum(done) == len(y):
            return adversarial_images
        adversarial_images.requires_grad =True

        out_adv = torch.sum(f(adversarial_images))
        out_adv.backward()

        g = adversarial_images.grad

        delta = x.to(device) - adversarial_images.to(device)
        delta = delta.to(device)

        map = torch.mul(g, delta).squeeze()

        # remove pixel from the allowed set, pay attetion to not removing pixels twice (remove only if no_change == 1)
        min_indices = []
        for i in range(no_change.shape[0]):
            #find first value where no_change != 0
            found = 0
            for row in range(len(no_change[i][0])):
                for column in range(len(no_change[i][0][row])):
                    if no_change[i][0][row][column] == 1:
                        min_idx = row,column
                        min_val = map[i][row][column]
                        found = 1
                    if found == 1:
                        break
                if found == 1:
                        break

            # find min pixel
            for row in range(map.shape[1]):
                for column in range(len(map[i][row])):
                    if map[i][row][column] < min_val:
                        if no_change[i][0][row][column] == 1:
                            min_val = map[i][row][column]
                            min_idx = row,column

            min_indices.append(min_idx)

        print("Removing pixels: "+str(min_indices))

        for i in range(no_change.shape[0]):
            row = min_indices[i][0]
            column = min_indices[i][1]
            no_change[i][0][row][column] = 0
        step += 1

        print("Number of L0 steps: "+str(step))
        adversarial_images.grad.zero_()


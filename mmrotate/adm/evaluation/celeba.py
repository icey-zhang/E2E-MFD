from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F


class CelebAEvaluator:
    @staticmethod
    def evaluate(model, data_loader, device):
        model.eval()
        with torch.no_grad():

            correct = np.zeros(40)
            total = len(data_loader.dataset)

            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                hrepr = model["encoder"](data)

                for i in range(40):
                    pi, ti = model[str(i)](hrepr), target[:, i]

                    output = pi.argmax(dim=1, keepdim=True)
                    correct[i] += output.eq(ti.view_as(output)).sum().item()

            accuracy = correct.sum() / (40 * total)
            accuracy_per_class = correct / total

            print(f"Acc = {accuracy}")

            return accuracy, accuracy_per_class

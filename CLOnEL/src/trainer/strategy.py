from collections import defaultdict
import torch


class Naive:
    def __init__(self, model):
        self.model = model

        self.use_cuda = torch.cuda.is_available()

    def before_backward(self, exp_counter, loss):
        return loss

    def after_training_exp(self, exp_counter, dataloader, criterion, optimizer):
        pass


class EWC:
    def __init__(self, model, ewc_lambda, mode, decay_factor=None):
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.mode = mode
        self.decay_factor = decay_factor

        self.use_cuda = torch.cuda.is_available()

        self.saved_params = defaultdict(dict)
        self.importances = defaultdict(dict)

    def compute_importances(self, dataloader, criterion, optimizer):

        importances = {}
        for name, param in self.model.named_parameters():
            importances[name] = torch.zeros_like(param)

        self.model.eval()

        for batch in dataloader:
            optimizer.zero_grad()

            batch_x, batch_y = batch

            outputs = self.model(batch_x)

            loss = criterion(outputs.float(), batch_y.to(outputs.device).float())
            loss.backward()

            for (k1, p), (k2, imp) in zip(
                    self.model.named_parameters(), importances.items()
            ):
                assert k1 == k2
                if p.grad is not None:
                    imp.data += p.grad.data.clone().pow(2)

        # average over mini batch length
        for _, imp in importances.items():
            imp.data /= float(len(dataloader))

        self.model.train()
        return importances

    def update_importances(self, importances, t):
        if self.mode == 'separate' or t == 0:
            self.importances[t] = importances
        elif self.mode == 'online':
            for name, curr_imp in importances.items():
                if name in self.importances[t - 1]:
                    old_imp = self.importances[t - 1][name]
                    self.importances[t][name] = self.decay_factor * old_imp + curr_imp
                else:
                    self.importances[t][name] = curr_imp

            if t > 0:
                del self.importances[t - 1]
        else:
            raise ValueError("Wrong EWC mode.")

    def before_backward(self, exp_counter, loss):
        if exp_counter == 0:
            return loss

        penalty = torch.tensor(0.0).to(next(self.model.parameters()).device)
        if self.mode == 'separate':
            for experience in range(exp_counter):
                for name, cur_param in self.model.named_parameters():
                    if name not in self.saved_params[experience]:
                        continue
                    saved_param = self.saved_params[experience][name]
                    imp = self.importances[experience][name]
                    penalty += (imp * (cur_param - saved_param) ** 2).sum()
        elif self.mode == 'online':
            for name, cur_param in self.model.named_parameters():
                if name not in self.saved_params[exp_counter - 1]:
                    continue
                saved_param = self.saved_params[exp_counter - 1][name]
                imp = self.importances[exp_counter - 1][name]
                penalty += (imp * (cur_param - saved_param) ** 2).sum()
        else:
            raise ValueError("Wrong EWC mode.")

        return loss + self.ewc_lambda * penalty

    def after_training_exp(self, exp_counter, dataloader, criterion, optimizer):
        importances = self.compute_importances(dataloader, criterion, optimizer)
        self.update_importances(importances, exp_counter)
        self.saved_params[exp_counter] = {name: param.clone() for name, param in self.model.named_parameters()}

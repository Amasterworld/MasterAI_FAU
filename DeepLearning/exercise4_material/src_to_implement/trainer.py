import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm
import numpy as np

class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()
            
    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))
    
    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])
        
    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})
            
    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients. By default, PyTorch accumulates (sums up) gradients when backward() is called. This behavior is not required here, so you need to ensure that all the gradients are zero before calling the backward.
        self._optim.zero_grad()
        # self._cuda is True, the input data x and labels y are moved to the GPU using the .cuda() method.
        if self._cuda:
            x = x.cuda()
            y = y.cuda()
        # -propagate through the network, forward pass
        # remember that we do not need self._mode.forward(x) because Pytorch used __call__ method
        outputs = self._model(x)
        # used the given loss function stored in self_crit to calculate loss
        loss = self._crit(outputs,  y)
        # -compute gradient by backward propagation
        loss.backward()
        # -update weights
        self._optim.step()
        # -return the loss
        return loss

    def val_test_step(self, x, y):
        if self._cuda:
            x = x.cuda()
            y = y.cuda()

        # Disable gradient calculation
        with t.no_grad():
            # Forward pass
            outputs = self._model(x)
            loss = self._crit(outputs, y)
            predictions = t.argmax(outputs, dim=1)

        return loss.item(), predictions

    def train_epoch(self):
        # set training mode
        # iterate through the training set
        # transfer the batch to "cuda()" -> the gpu if a gpu is given
        # perform a training step
        # calculate the average loss for the epoch and return it
        #TODO
        self._model.train()  # Set the model to training mode
        total_loss = 0.0
        num_batches = 0

        for x, y in self._train_dl:
            if self._cuda:
                x = x.cuda()
                y = y.cuda()

            self._optim.zero_grad()  # Reset gradients

            loss = self.train_step(x, y)  # Perform a training step

            loss.backward()  # Compute gradients
            self._optim.step()  # Update weights

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def val_test(self):
        # set eval mode. Some layers have different behaviors during training and testing (for example: Dropout, BatchNorm, etc.). To handle those properly, you'd want to call model.eval()
        # disable gradient computation. Since you don't need to update the weights during testing, gradients aren't required anymore.
        # iterate through the validation set
        # transfer the batch to the gpu if given
        # perform a validation step
        # save the predictions and the labels for each batch
        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        # return the loss and print the calculated metrics
        #TODO
        self._model.eval()  # Set the model to evaluation mode
        total_loss = 0.0
        num_batches = 0
        predictions_list = []
        labels_list = []

        with t.no_grad():  # Disable gradient computation
            for x, y in self._val_test_dl:
                if self._cuda:
                    x = x.cuda()
                    y = y.cuda()

                loss, predictions = self.val_test_step(x, y)  # Perform a validation/test step

                total_loss += loss.item()
                num_batches += 1
                predictions_list.append(predictions.cpu().numpy())
                labels_list.append(y.cpu().numpy())

        avg_loss = total_loss / num_batches

        # Concatenate predictions and labels for all batches
        predictions_all = np.concatenate(predictions_list)
        labels_all = np.concatenate(labels_list)

        # Calculate and print metrics of your choice
        accuracy = calculate_accuracy(predictions_all, labels_all)
        precision, recall, f1_score = calculate_precision_recall_f1(predictions_all, labels_all)
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")

        return avg_loss

    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch
        # TODO

        train_losses = []
        val_losses = []
        epoch = 0
        best_val_loss = float('inf')
        while True:
            # stop by epoch number
            # train for a epoch and then calculate the loss and metrics on the validation set
            # append the losses to the respective lists
            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            # check whether early stopping should be performed using the early stopping criterion and stop if so
            # return the losses for both training and validation
            # stop by epoch number
            if epochs > 0 and epoch >= epochs:
                break
            # train for a epoch and then calculate the loss and metrics on the validation set
            train_loss = self.train_epoch()
            val_loss, val_metrics = self.evaluate()
            # append the losses to the respective lists
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint()
            # check whether early stopping should be performed using the early stopping criterion and stop if so
            if self.early_stopping_criterion(val_losses):
                break
            # increment the epoch counter
            epoch += 1
        # return the losses for both training and validation
        return train_losses, val_losses

                    
        
        
        

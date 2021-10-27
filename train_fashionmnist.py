# Imports

import torch
import torch.nn as nn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from model_fashionmnist import FashionCNN

import wandb

wandb.init(project='nonconvex-FashionMNIST', entity='iol')

use_gpu = torch.cuda.is_available()
if use_gpu:
        print('Using GPU.')

# Download test data

# Do not set shuffle if want to reproduce the results
shuffle = True #param {type: "boolean"}

# Batch size
b_size  = 100 #param {type: "integer"}

# Number of batches
n_batches = 1 #param {type: "integer"}

test_set = torchvision.datasets.FashionMNIST("./data", download=True, train=False, transform=transforms.Compose([transforms.ToTensor()]))

test_loader = torch.utils.data.DataLoader(test_set, batch_size=b_size, shuffle=shuffle, drop_last=True)

# Load the model

pretrained_model = True #param {type: "boolean"}
save_model = False #param {type: "boolean"}
model_label = "v1" #param {type: "string"}

if not pretrained_model:

    train_set = torchvision.datasets.FashionMNIST("./data", download=True, transform=
    transforms.Compose([transforms.ToTensor()]))
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=b_size, drop_last=True)
    # Make a model of CNN class

    model = FashionCNN()

    if use_gpu:
        model = model.cuda()

    error = nn.CrossEntropyLoss()

    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # print(model)

    # Training a network and testing it on test dataset

    num_epochs = 5
    count = 0
    # Lists for visualization of loss and accuracy
    loss_list = []
    iteration_list = []
    accuracy_list = []

    # Lists for knowing class-wise accuracy
    predictions_list = []
    labels_list = []

    for epoch in range(num_epochs):
        for images, labels in train_loader:
            # Transfering images and labels to GPU if available
            if use_gpu:
                images = images.cuda()
                labels = labels.cuda()

            train = Variable(images.view(b_size, 1, 28, 28))
            labels = Variable(labels)

            # Forward pass
            outputs = model(train)
            loss = error(outputs, labels)

            # Initializing a gradient as 0 so there is no mixing of gradient among the batches
            optimizer.zero_grad()

            # Propagating the error backward
            loss.backward()

            # Optimizing the parameters
            optimizer.step()

            count += 1

            # Testing the model

            if not (count % 50):  # It's same as "if count % 50 == 0"
                total = 0
                correct = 0

                for images, labels in test_loader:
                    if use_gpu:
                        images = images.cuda()
                        labels = labels.cuda()

                    labels_list.append(labels)

                    test = Variable(images.view(b_size, 1, 28, 28))

                    outputs = model(test)

                    predictions = torch.max(outputs, 1)[1]
                    if use_gpu:
                        predictions = predictions.cuda()
                    predictions_list.append(predictions)
                    correct += (predictions == labels).sum()

                    total += len(labels)

                accuracy = correct * 100 / total
                loss_list.append(loss.data)
                iteration_list.append(count)
                accuracy_list.append(accuracy)

            if not (count % 500):
                print("Iteration: {}, Loss: {}, Accuracy: {}%".format(count, loss.data, accuracy))

    if save_model:
        torch.save(model, "model")

        run = wandb.init(job_type="model-creation")
        artifact = wandb.Artifact('pretrained-model' + model_label, type='model')
        artifact.add_file("model")
        run.log_artifact(artifact)

else:
    run = wandb.init(job_type="model-training")
    artifact = run.use_artifact('pretrained-modelv1:latest')
    artifact_dir = artifact.download()

    print(artifact_dir)

    if use_gpu:
        model = torch.load(artifact_dir+"/model")
    else:
        model = torch.load(artifact_dir + "/model", map_location=torch.device('cpu'))
    model.eval()

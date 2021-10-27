import wandb
import torch
import argparse

from model_fashionmnist import FashionCNN
from train_fashionmnist import test_loader
from CW_L2 import carlini_wagner_l2
from CW_L0 import carlini_wagner_l0


use_gpu = torch.cuda.is_available()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Define hyperparameters.')
    parser.add_argument('--dataset', type=str, default='f_mnist', help='cifar10, mnist, f_mnist')
    parser.add_argument('--attack', type=str, default='carlini_wagner_l2',
                        help='carlini_wagner_l2, carlini_wagner_l0')
    parser.add_argument('--path_results', type=str, default='none')
    parser.add_argument('--n_examples', type=int, default=50)
    parser.add_argument('--data_dir', type=str, default='./data')

    hps = parser.parse_args([])

    run = wandb.init(job_type="model-training")
    artifact = run.use_artifact('pretrained-modelv1:v2')
    artifact_dir = artifact.download()

    print(artifact_dir)

    if use_gpu:
        model = torch.load(artifact_dir + "/model")
    else:
        model = torch.load(artifact_dir + "/model", map_location=torch.device('cpu'))
    model.eval()

    if hps.attack == 'carlini_wagner_l2':
        for data in test_loader:
            images, labels = data
            no_change = torch.ones_like(images)
            if use_gpu:
                images = images.cuda()
                labels = labels.cuda()
                no_change = no_change.cuda()
            # no_change = torch.randint(0,2,(images.shape))
            # print(no_change)
            #no_change = no_change.to(device)
            out_img = adversarial_images_l2 = carlini_wagner_l2(model_fn=model, ox = images, x=images, n_classes=10,
                                                                           no_change=no_change)
            predictions = torch.argmax(model(out_img), 1)
            # print(out_img)
            print("True labels:" + str(labels) + ", predictions: " + str(predictions))

            break

    elif hps.attack == 'carlini_wagner_l0':
        for data in test_loader:
            images, labels = data
            # no_change = torch.randint(0,2,images.shape)
            #no_change = torch.ones_like(images)
            if use_gpu:
                images = images.cuda()
                labels = labels.cuda()
                #no_change = no_change.cuda()
            '''
            no_change[0][0][1][12]= 1
            no_change[0][0][3][14]= 1
            no_change[0][0][15][1]= 1
            no_change[1][0][22][8]= 1
            no_change[1][0][12][12]= 1
            no_change[1][0][7][15]= 1
            no_change[2][0][3][1]= 1
            no_change[2][0][13][18]= 1
            no_change[2][0][25][16]= 1
            no_change[3][0][11][14]= 1
            no_change[3][0][12][23]= 1
            no_change[3][0][24][25]= 1
            print(no_change)
            '''
            #no_change = no_change.to(device)
            out_img = adversarial_images_l2 = carlini_wagner_l0(model_fn=model, x=images, n_classes=10)
            predictions = torch.argmax(model(out_img), 1)
            print("True labels: " + str(labels) + ", predictions: " + str(predictions))

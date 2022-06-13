from __future__ import print_function
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from models import *
from six.moves import urllib
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
import time
np.random.seed(8)
random.seed(8)
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

# settings
pretrained_model = "./data/lenet_mnist_model.pth"
use_cuda = True
print(torch.cuda.device_count())

# hyper parameters
target_label = 0
samples_size = 45
epsilons = [0, .05, .1, .15, .2, .25, .3]
loss= "targeted"

print("CUDA Available: ", torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# main
test_data = datasets.MNIST('./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

# MNIST Test dataset and dataloader declaration
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)
lenet_model = LeNet().to(device)
# Load the pretrained lenet_model
lenet_model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))
lenet_model.eval()

accuracies = []
examples = []

# inception model
googlenet_model = models.googlenet(pretrained=True)
googlenet_model.to(device)
googlenet_model.eval()
flatten_return_nodes = {
    'flatten': 'high-level-features',
}
inception4c_return_nodes = {
    'inception4c.cat': 'mid-level-features',
}
inception3a_return_nodes = {
    'inception3a.cat': 'low-level-features'
}
# move these to device
googlenet_flatten_feature_extractor = create_feature_extractor(googlenet_model, flatten_return_nodes)

googlenet_flatten_feature_extractor = googlenet_flatten_feature_extractor.to(device)

googlenet_inception4c_feature_extractor = create_feature_extractor(googlenet_model, inception4c_return_nodes)
googlenet_inception4c_feature_extractor = googlenet_inception4c_feature_extractor.to(device)

googlenet_inception3a_feature_extractor = create_feature_extractor(googlenet_model, inception3a_return_nodes)
googlenet_inception3a_feature_extractor = googlenet_inception3a_feature_extractor.to(device)

# this is to use the same set of samples for testing across different epsilon trials.
image_files = get_ImageNet_test_data()
# Run test for each epsilon
for eps in epsilons:
    start_time = time.time()
    acc, ex = test_attack_MNIST(lenet_model, device, test_loader, eps, samples_size, loss, 7)
    #acc, ex = test_attack_ImageNet(googlenet_model, googlenet_flatten_feature_extractor, device, eps, loss, image_files, 292)
    #acc, ex = test_attack_ImageNet_FGSM(googlenet_model, device, eps, image_files)
    print("--- %s minutes taken to execute a single epsilon---" % ((time.time() - start_time)/60))
    accuracies.append(acc)
    examples.append(ex)


# plot epsilons variations against accuracy
plt.figure(figsize=(5, 5))
plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .35, step=0.05))
plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
#plt.savefig("epsilons for " + loss + " targeted attack on -MNIST.png")
cnt = 0
plt.figure(figsize=(8, 10))
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(epsilons), len(examples[0]), cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
        orig, adv, ex = examples[i][j]
        plt.title("{} -> {}".format(orig, adv), fontdict = {'fontsize' : 8})
        plt.imshow(ex, cmap="gray")
plt.tight_layout()
#plt.savefig("examples of targeted attack -MNIST.png")
plt.show()

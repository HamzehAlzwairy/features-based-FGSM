import torch
import torch.nn.functional as F
from torch import linalg as LA
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import glob
import json
import random
import time

np.random.seed(8)
random.seed(8)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def attack_loss(image_features, target_mean):
    dist = image_features - target_mean
    loss = LA.norm(dist)
    return loss


def features_fgsm_attack(image, epsilon, data_grad, chosen_loss, imagenet=False):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    if chosen_loss == "targeted":
        perturbed_image = image - epsilon * sign_data_grad
    else:
        perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    if not imagenet:
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def fgsm_attack(image, epsilon, data_grad, imagenet=False):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    if not imagenet:
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def test_attack_MNIST(model, device, test_loader, epsilon, samples_size, chosen_loss, target_class=None):
    correct = 0
    adv_examples = []
    MNIST_mean_vectors = get_MNIST_mean_vectors(model, samples_size, device)
    n = len(test_loader)
    for source_image, source_label in test_loader:
        if chosen_loss == "targeted":
            target_features_mean = MNIST_mean_vectors[target_class]
            if source_label.item() == target_class:  # this means target is same as source, so just ignore
                n = n - 1
                continue
        else:
            target_features_mean = MNIST_mean_vectors[source_label.item()]

        source_image, source_label = source_image.to(device), source_label.to(device)

        source_image.requires_grad = True

        output, source_image_features = model(source_image, True)
        init_pred = output.max(1, keepdim=True)[1]
        if init_pred.item() != source_label.item():
            n = n - 1
            continue

        loss = attack_loss(source_image_features.squeeze(), target_features_mean)
        model.zero_grad()

        # The first time we input a mean vector, the differentiation of it can be seen .. but during that backward
        # all the intermediary results are deleted when they are not needed anymore, so when we call backward on the
        # same mean vector again, it can't see where it came from or its graph.
        loss.backward(retain_graph=True)
        source_image_grad = source_image.grad.data

        # Call FGSM Attack
        perturbed_source_image = features_fgsm_attack(source_image, epsilon, source_image_grad, chosen_loss)
        # Re-classify the perturbed image
        output = model(perturbed_source_image)
        final_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        if final_pred.item() == source_label.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_source_image.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_source_image.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

    # Calculate final accuracy for this epsilon
    final_acc = correct / float(n)
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, n, final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples


def get_MNIST_target_batch(target_label, sample_size):
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
    ]))
    indices_of_target = train_data.targets == target_label
    target_images = train_data.data[indices_of_target]
    target_images = target_images.type(torch.float)
    # sample sample_size random images from the set of target images
    target_samples = target_images[torch.randint(len(target_images), (sample_size,))]
    # because the conv layer has out * d * k * k kernel size, c = 1 and here we need to add that dimen for compaitability
    target_samples = torch.unsqueeze(target_samples, 1)
    return target_samples


def get_MNIST_mean_vectors(model, samples_size, device):
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    mean_vectors = {}
    for c in classes:
        # get a batch of target images to be used for creating mean feature vector
        target_batch = get_MNIST_target_batch(c, samples_size)
        # this can be moved out of the loop, just make sure to move it to GPU too.
        target_batch = target_batch.to(device)
        target_outputs, target_features = model(target_batch, True)  # features = batch_size x 320
        target_features_mean = torch.mean(target_features, 0, dtype=torch.float)  # 320
        mean_vectors[c] = target_features_mean
    return mean_vectors


def get_ImageNet_batch(c, used_images):
    images = []
    classes_dictionary = json.load(open("./data/imagenet_class_index.json"))
    for k, v in classes_dictionary.items():
        if c == int(k):
            parent_folder = v[0]
            break
    files = glob.glob("./data/val_imagenet/" + parent_folder + "/*.JPEG")
    for file in files:
        try:
            # used_images are the images that are not used to sample the test data
            # this is to ensure that any result obtained is not due to bias
            if file not in used_images:
                images.append(preprocess(Image.open(open(file, "rb"))))
        except:
            continue  # 1 channel image causes error
    batch = torch.stack(images, 0)

    # print("target samples retrieved from imagenet dataset have shape {0} .. should have 3 channels like B x C x W x H".format(batch.shape))
    return batch


def get_ImageNet_mean_vector(feature_extractor, device, target_class, used_images):
    batch = get_ImageNet_batch(target_class, used_images)
    batch = batch.to(device)
    features_batch = feature_extractor(batch)
    features_batch = list(features_batch.values())[0]  # batch_size x 1024
    # if the feature extractor is not a flatten layer, we receive 4D features and we need to flatten
    if len(features_batch.shape) > 2:
        features_batch = torch.flatten(features_batch, 1)
    # need flatten if not flatten extractor
    # print("features returned from extractor have the shape {0} need to flatten here before mean so check output shape then make it B x F".format(features_batch.shape))
    mean_vector = torch.mean(features_batch, 0, dtype=torch.float)
    return mean_vector


def test_attack_ImageNet(model, feature_extractor, device, epsilon, chosen_loss, image_files, target_class=None):
    classes_dictionary = json.load(open("./data/imagenet_class_index.json"))
    correct = 0
    adv_examples = []
    retrieved = False
    # get shuffled image file paths
    n = len(image_files)
    used_images = list(image_files.keys())
    for image_file, source_label in image_files.items():
        source_label = torch.Tensor([source_label])
        img = Image.open(image_file)
        try:
            source_image = preprocess(img)
        except:
            # these images are not 3 channels but rather black and white ( 1 channel ), so ignore them
            n = n - 1
            continue
        if chosen_loss == "targeted":
            if source_label == target_class:  # this means target is same as source, so just ignore
                n = n - 1
                continue
            if not retrieved:
                target_features_mean = get_ImageNet_mean_vector(feature_extractor, device, target_class,
                                                                used_images)  # 1024
                retrieved = True
        else:  # untargeted attack
            target_features_mean = get_ImageNet_mean_vector(feature_extractor, device, source_label, used_images)

        source_image, source_label = source_image.to(device), source_label.to(device)

        source_image = source_image.unsqueeze(0)
        source_image.requires_grad = True
        output = model(source_image)

        # print("Output shape from model given input image is {0} ".format(output.shape))
        # googlenet might be outputing raw neuron outputs, therefore we apply softmax
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        # print("Probabilities shape from model given input image is {0} and a sample is".format(probabilities.shape))
        init_pred = probabilities.max(0, keepdim=True)[1]
        if init_pred.item() != source_label.item():
            n = n - 1
            continue
        source_image_features = feature_extractor(source_image)
        source_image_features = list(source_image_features.values())[0]
        loss = attack_loss(source_image_features.squeeze(), target_features_mean)
        model.zero_grad()
        loss.backward(retain_graph=True)

        source_image_grad = source_image.grad.data
        # Call FGSM Attack
        perturbed_source_image = features_fgsm_attack(source_image, epsilon, source_image_grad, chosen_loss, True)

        output = model(perturbed_source_image)
        output_probabilities = torch.nn.functional.log_softmax(output[0], dim=0)
        final_pred = output_probabilities.max(0, keepdim=True)[1]  # get the index of the max log-probability
        if final_pred.item() == source_label.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_source_image.squeeze().detach().cpu()
                adv_examples.append((classes_dictionary[str(init_pred.item())][1],
                                     classes_dictionary[str(final_pred.item())][1], adv_ex))
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_source_image.squeeze().detach().cpu()
                adv_examples.append((classes_dictionary[str(init_pred.item())][1],
                                     classes_dictionary[str(final_pred.item())][1], adv_ex))
    # Calculate final accuracy for this epsilon
    final_acc = np.round_(correct / float(n), 4)
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, n, final_acc))
    del target_features_mean
    # Return the accuracy and an adversarial example
    return final_acc, adv_examples


def test_attack_ImageNet_FGSM(model, device, epsilon, image_files):
    classes_dictionary = json.load(open("./data/imagenet_class_index.json"))
    correct = 0
    adv_examples = []
    # get shuffled image file paths
    n = len(image_files)
    for image_file, source_label in image_files.items():
        source_label = torch.Tensor([source_label])
        img = Image.open(image_file)
        try:
            source_image = preprocess(img)
        except:
            # these images are not 3 channels but rather black and white ( 1 channel ), so ignore them
            n = n - 1
            continue

        source_image, source_label = source_image.to(device), source_label.to(device)

        source_image = source_image.unsqueeze(0)
        source_image.requires_grad = True
        output = model(source_image)

        # print("Output shape from model given input image is {0} ".format(output.shape))
        # googlenet might be outputing raw neuron outputs, therefore we apply softmax
        probabilities = torch.nn.functional.log_softmax(output[0], dim=0)
        # print("Probabilities shape from model given input image is {0} and a sample is".format(probabilities.shape))
        init_pred = probabilities.max(0, keepdim=True)[1]
        if init_pred.item() != source_label.item():
            n = n - 1
            continue
        loss = F.nll_loss(probabilities.unsqueeze(0), source_label.type(torch.LongTensor).to(device))
        model.zero_grad()
        loss.backward()

        source_image_grad = source_image.grad.data
        # Call FGSM Attack
        perturbed_source_image = fgsm_attack(source_image, epsilon, source_image_grad, True)

        output = model(perturbed_source_image)
        output_probabilities = torch.nn.functional.log_softmax(output[0], dim=0)
        final_pred = output_probabilities.max(0, keepdim=True)[1]  # get the index of the max log-probability
        if final_pred.item() == source_label.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_source_image.squeeze().detach().cpu()
                adv_examples.append((classes_dictionary[str(init_pred.item())][1],
                                     classes_dictionary[str(final_pred.item())][1], adv_ex))
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_source_image.squeeze().detach().cpu()
                adv_examples.append((classes_dictionary[str(init_pred.item())][1],
                                     classes_dictionary[str(final_pred.item())][1], adv_ex))
    # Calculate final accuracy for this epsilon
    final_acc = np.round_(correct / float(n), 4)
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, n, final_acc))
    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

def get_ImageNet_test_data():
    image_files = {}
    classes_dictionary = json.load(open("./data/imagenet_class_index.json"))
    for k, v in classes_dictionary.items():
        label = int(k)
        parent_folder = v[0]
        files = glob.glob("./data/val_imagenet/" + parent_folder + "/*.JPEG")
        np.random.shuffle(files)
        files = files[:5]
        for file in files:
            image_files[file] = label
    # shuffling images
    l = list(image_files.items())
    np.random.shuffle(l)
    image_files = dict(l)
    return image_files

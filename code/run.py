"""Implementation of sample attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import csv
import random
import shutil
import time
import warnings
from collections import OrderedDict
import glob as glob

start_time = time.time()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as functional
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from PIL import Image
import timm
from multiprocessing.dummy import Pool as ThreadPool

parser = argparse.ArgumentParser('Running script', add_help=False)
parser.add_argument('--input_dir', default='../input_dir', type=str)
parser.add_argument('--output_dir', default='../output_dir', type=str)

parser.add_argument('--batch_size', default=4, type=int, metavar='N',
                    help='How many images process at one time.')
parser.add_argument('--image_width', default=500, type=int, metavar='N',
                    help='Width of each input images.')
parser.add_argument('--image_height', default=500, type=int, metavar='N',
                    help='Height of each input images.')
parser.add_argument('--image_resize', default=530, type=int, metavar='N',
                    help='Height of each input images.')
parser.add_argument('--num_iter', default=80, type=int, metavar='N',
                    help='Number of iterations.')
parser.add_argument('--momentum', default=1.0, type=float, metavar='M',
                    help='Momentum.')
parser.add_argument('--max_epsilon', default=16.0, type=float, metavar='M',
                    help='Maximum size of adversarial perturbation.')
parser.add_argument('--prob', default=0.7, type=float, metavar='M',
                    help='probability of using diverse inputs.')


FLAGS = parser.parse_args()

main_device='cuda:0'

def normalize(array, mean, std):
    array = array.transpose(2, 0, 1)
    for i in range(3):
        array[i, :, :] = (array[i, :, :] - mean[i]) / std[i]
    return array

def load_images(input_dir, batch_shape):
    """Read png images from input directory in batches.

    Args:
        input_dir: input directory
        batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

    Yields:
        filenames: list file names without path of each image
            Lenght of this list could be less than batch_size, in this case only
            first few images of the result are elements of the minibatch.
        images: array with all images from this batch
    """
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    labels = np.zeros([batch_size], np.int32)

    with open(os.path.join(FLAGS.input_dir, 'dev.csv'), encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in list(reader):
            filepath = os.path.join(input_dir, row[0])
            image = np.array(Image.open(filepath)).astype(np.float) / 255.0

            images[idx, :, :, :] = image.transpose(2, 0, 1)
            labels[idx] = row[1]
            
            filenames.append(os.path.basename(filepath))
            idx += 1
            if idx == batch_size:
                yield filenames, images, labels
                filenames = []
                images = np.zeros(batch_shape)
                labels = np.zeros([batch_size], np.int32)
                idx = 0
        if idx > 0:
            yield filenames, images, labels


def save_images(images, filenames, output_dir):
    """Saves images to the output directory.

    Args:
        images: array with minibatch of images
        filenames: list of filenames without path
            If number of file names in this list less than number of images in
            the minibatch then only first len(filenames) images will be saved.
        output_dir: directory where to save images
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, filename in enumerate(filenames):
        Image.fromarray(np.uint8((images[i, :, :, :]) * 255)).save(os.path.join(output_dir, filename))

def clip_by_tensor(t,t_min,t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    t=t.float()
    t_min=t_min.float()
    t_max=t_max.float()
 
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result


def input_diversity(images):
    if (np.random.uniform(0, 1) < FLAGS.prob):
        rnd = int(np.random.uniform(FLAGS.image_width, FLAGS.image_resize))
        rescaled = F.interpolate(images, [rnd, rnd], mode='bicubic')
        h_rem = FLAGS.image_resize - rnd
        w_rem = FLAGS.image_resize - rnd
        pad_top = int(np.random.uniform(0, h_rem))
        pad_bottom = h_rem - pad_top
        pad_left = int(np.random.uniform(0, w_rem))
        pad_right = w_rem - pad_left
        padded = F.pad(rescaled, (pad_left,pad_right,pad_top,pad_bottom))
        return padded
    else:
        return images


def CW_loss(logits, label, c=99999.0):
    batch_size = logits.size(0)
    class_num = logits.size(1)
    logits_mask = torch.zeros(batch_size, class_num).to(main_device).scatter_(1, label.unsqueeze(-1), 1)
    logits_this = torch.sum(logits_mask * logits, axis=-1)
    logits_that = torch.max(logits - c * logits_mask, axis=-1)[0]
    return (logits_that-logits_this).sum()

def main():
    eps = FLAGS.max_epsilon / 255.0
    num_iter = FLAGS.num_iter
    alpha = eps / num_iter
    momentum = FLAGS.momentum
    num_classes = 1000
    batch_shape = [FLAGS.batch_size, 3, FLAGS.image_height, FLAGS.image_width]

    # model selection
    main_device='cuda:0'
    model_names=['tf_efficientnet_b6_ns', 'resnest269e', 'vit_large_patch16_384', 'tf_efficientnet_b6_ap']
    models=[]
    for i, model_name in enumerate(model_names):
        model=timm.create_model(model_name,num_classes=1000,pretrained=True)
        model=model.to('cuda:{}'.format(i))
        model.eval()
        models.append(model)

    def forward_map_func(args_dict, q):
        # with global variables mdoels and main_device
        i = args_dict['id']
        x_adv = args_dict['tensor']
        diverse_input = input_diversity(x_adv)
        image_resize = models[i].default_cfg['input_size'][1:]
        mean = torch.tensor(models[i].default_cfg['mean']).view(3,1,1).to(main_device)
        std = torch.tensor(models[i].default_cfg['std']).view(3,1,1).to(main_device)
        resized_tensor=F.interpolate(diverse_input,size=image_resize,mode='bicubic')
        gaussian_tensor=functional.gaussian_blur(resized_tensor,kernel_size=(5,5),sigma=(0.9,0.9))
        normalized_tensor=(gaussian_tensor-mean)/std
        output = models[i](normalized_tensor.to('cuda:{}'.format(i))).to(main_device)
        q.put(output)

    print(time.time() - start_time)

    for filenames, images, labels in load_images(os.path.join(FLAGS.input_dir, 'images'), batch_shape):
        batch_start = time.time()
        input_image = torch.from_numpy(images).float().to(main_device)
        input_tensor = input_image.to(main_device)

        print(labels)
        target = torch.from_numpy(labels).long().to(main_device)

        grad = torch.zeros_like(input_tensor)
        clip_tensor_one = torch.ones_like(input_tensor)
        clip_tensor_zero = torch.zeros_like(input_tensor)
        x_max = clip_by_tensor(input_tensor + eps, clip_tensor_zero, clip_tensor_one)
        x_min = clip_by_tensor(input_tensor - eps, clip_tensor_zero, clip_tensor_one)
        x_adv = input_tensor
        for i in range(FLAGS.num_iter):
            x_adv = x_adv.clone().detach().requires_grad_(True)

            import threading
            from queue import Queue
            q=Queue()
            threads = []
            for d in range(len(model_names)):
                t = threading.Thread(target=forward_map_func, args=({'id': d, 'tensor': x_adv}, q))
                threads.append(t)
                t.start()

            [thread.join() for thread in threads]
            logits=torch.zeros((FLAGS.batch_size,num_classes)).to(main_device)
            for index in range(len(threads)):
                logits += q.get().to(main_device)/len(models)

            # loss = F.cross_entropy(logits, target)
            loss = CW_loss(logits, target)
            loss.backward()
            noise = x_adv.grad
            noise = noise / torch.mean(torch.abs(noise), dim=[1,2,3], keepdim=True)
            noise = momentum * grad + noise
            x_adv = x_adv + alpha * torch.sign(noise)
            x_adv = clip_by_tensor(x_adv, x_min, x_max)
            grad = noise
        save_images(x_adv.permute(0,2,3,1).detach().cpu().numpy(), filenames, FLAGS.output_dir)
        print(time.time()-batch_start)

if __name__ == '__main__':
    main()

"""Training a face recognizer with TensorFlow based on the FaceNet paper
FaceNet: A Unified Embedding for Face Recognition and Clustering: http://arxiv.org/abs/1503.03832
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import sys
import tensorflow as tf
import numpy as np
import importlib
import itertools
import argparse
import facenet
import lfw

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import resnet
from tqdm import tqdm
from matplotlib import pyplot as plt

from tensorflow.python.ops import data_flow_ops

from six.moves import xrange  # @UnresolvedImport

def main(args):
    #network = importlib.import_module(args.model_def)

    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)

    # Write arguments to a text file
    #facenet.write_arguments_to_file(args, os.path.join(log_dir, 'arguments.txt'))
    
    # Store some git revision info in a text file in the log directory
    src_path,_ = os.path.split(os.path.realpath(__file__))
    #facenet.store_revision_info(src_path, log_dir, ' '.join(sys.argv))

    np.random.seed(seed=args.seed)
    #train_set = facenet.get_dataset(args.data_dir)
    
    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)
    if args.pretrained_model:
        print('Pre-trained model: %s' % os.path.expanduser(args.pretrained_model))
    
    if args.lfw_dir:
        print('LFW directory: %s' % args.lfw_dir)
        # Read the file containing the pairs used for testing
        pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))
        # Get the paths for the corresponding images
        lfw_paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs)
    
    # TORCH TRAINING - RAVIT
    model = resnet.resnet18(full=False)
    modelName = "ResNet18"

    trainModel(model, modelName, args.data_dir, args.lfw_dir, model_dir, log_dir)

    return model_dir

def trainModel(model, modelName, data_dir, lfw_dir, models_base_dir, log_dir, learning_rate=0.05, epochs=100, alpha=0.2, checkpoint_freq=10):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_set = facenet.FaceRecognitionDataset(data_dir, model=model, alpha=alpha)
    test_set = facenet.FaceRecognitionDataset(lfw_dir, alpha=alphs)

    dummy_input = torch.ones((1, 3, train_set.image_dim, train_set.image_dim))

    log_file = os.path.join(log_dir, "log.txt")
    model_ckpt_best = os.path.join(models_base_dir, modelName + "_best")
    model_ckpt_recent = os.path.join(models_base_dir, modelName + "_recent")

    with open(log_file, "a") as f:
        f.write("Begin training\n")

    training_history = {"train_loss":[], "train_acc":[], "eval_loss":[], "eval_acc":[]}
    best_eval_accuracy = 0
    for epoch in tqdm(range(epochs)):
        if epoch % checkpoint_freq == 0:
            torch.save(model.state_dict(), model_ckpt_recent + ".pt")
            torch.onnx.export(model, dummy_input, model_ckpt_recent + ".onnx", opset_version=12)

        epoch_start = datetime.now()
        loss, accuracy = train(epoch, train_set, model, alpha, optimizer, log_file)

        eval_loss, eval_accuracy = evaluate(test_set, model, alpha)

        if eval_accuracy > best_eval_accuracy:
            torch.save(model.state_dict(), model_ckpt_best + ".pt")
            torch.onnx.export(model, dummy_input, model_ckpt_best + ".onnx", opset_version=12)
            best_eval_accuracy = eval_accuracy

        epoch_end = datetime.now()

        training_history["train_loss"].append(loss)
        training_history["train_acc"].append(accuracy)
        training_history["eval_loss"].append(eval_loss)
        training_history["eval_acc"].append(eval_accuracy)
        

        epoch_string = "Epoch[{:d}] finished in {:s} - Loss: {:.3f}, Acuracy: {:.3f}, Eval Loss: {:.3f}, Eval Accuracy {:.3f}".format(epoch+1, 
            str(epoch_end-epoch_start), loss, accuracy, eval_loss, eval_accuracy)
        print(epoch_string)
        with open(log_file, "a") as f:
            f.write(epoch_string + "\n")
    
    torch.save(model.state_dict(), os.path.join(models_base_dir, modelName + ".pt"))
    torch.onnx.export(model, dummy_input, os.path.join(models_base_dir, modelName+".onnx"), opset_version=9)

    epochs_list = list(range(epochs))

    plt.clf()
    plt.plot(epochs_list, training_history["train_loss"])
    plt.plot(epochs_list, training_history["eval_loss"])
    plt.legend(["train_loss", "eval_loss"])
    plt.yscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(log_dir, modelName + "_loss.png"))

    plt.clf()
    plt.plot(epochs_list, training_history["train_acc"])
    plt.plot(epochs_list, training_history["eval_acc"])
    plt.legend(["train_acc", "eval_acc"])
    plt.xlabel("Epochs")
    plt.ylabel("Acuracy")
    plt.savefig(os.path.join(log_dir, modelName + "_acc.png"))

def train(epoch, train_set, model, alpha, optimizer, log_file):
    loss_meter = facenet.AverageMeter()
    accuracy_meter = facenet.AverageMeter()
    batch_number = 0
    epoch_size = train_set.epoch_size
    for triplets in iter(train_set):
        nrof_triplets = len(triplets)
        nrof_batches = int(np.ceil(nrof_triplets/train_set.batch_size))
        i = 0
        while i < nrof_batches:
            i += 1
            batch_number += 1
            #batch_size = min(nrof_triplets-i*train_set.batch_size, train_set.batch_size)

            batch_start = datetime.now()
            batch = triplets[(i*train_set.batch_size):(i*train_set.batch_size)+train_set.batch_size]
            batch_size = len(batch)

            triplets_f = np.array(batch).flatten()
            batch_tensor = np.reshape(
                train_set.image_paths_to_images(triplets_f), (-1, 3, 3, train_set.image_dim, train_set.image_dim))
            batch_tensor = torch.tensor(batch_tensor)

            # calculate embeddings for anchors
            anchor_images = batch_tensor[:,0]
            positive_images = batch_tensor[:,1]
            negative_images = batch_tensor[:,2]

            try:
                anchor_embedding = model(anchor_images)
                with torch.no_grad():
                    positive_embedding = model(positive_images)
                    negative_embedding = model(negative_images)
            except Exception:
                print("Encountered exception in forward pass. batch_tensor has shape", batch_tensor.shape)
                break
            
            loss = facenet.triplet_loss(anchor_embedding, positive_embedding, negative_embedding, alpha)
            _, _, accuracy = facenet.calculate_accuracy(anchor_embedding, positive_embedding, negative_embedding, alpha)

            loss_meter.update(loss, batch_size)
            accuracy_meter.update(accuracy, batch_size)

            # optimizer and learning rate step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_end = datetime.now()
            
            batch_str = "\tEpoch [{:d}] {:d}/{:d} ({:d} triplets) finished in {:s}. Loss: {:.3f}. Accuracy: {:.3f}\n".format(epoch, batch_number, epoch_size, batch_size,
                str(batch_end-batch_start), float(loss_meter.val), float(accuracy_meter.val))
            print(batch_str[:-1])
            with open(log_file, "a") as f:
                f.write(batch_str)
    return loss_meter.val, accuracy_meter.val

def evaluate(test_set, model, alpha):
    loss_meter = facenet.AverageMeter()
    accuracy_meter = facenet.AverageMeter()
    batch_number = 0
    epoch_size = test_set.epoch_size
    for triplets in iter(test_set):
        nrof_triplets = len(triplets)
        nrof_batches = int(np.ceil(nrof_triplets/test_set.batch_size))
        i = 0
        while i < nrof_batches:
            i += 1
            batch_number += 1

            batch_start = datetime.now()
            batch = triplets[(i*test_set.batch_size):(i*test_set.batch_size)+test_set.batch_size]
            batch_size = len(batch)

            triplets_f = np.array(batch).flatten()
            batch_tensor = np.reshape(
                test_set.image_paths_to_images(triplets_f), (-1, 3, 3, test_set.image_dim, test_set.image_dim))
            batch_tensor = torch.tensor(batch_tensor)

            # calculate embeddings for anchors
            anchor_images = batch_tensor[:,0]
            positive_images = batch_tensor[:,1]
            negative_images = batch_tensor[:,2]

            try:
                with torch.no_grad():
                    anchor_embedding = model(anchor_images)
                    positive_embedding = model(positive_images)
                    negative_embedding = model(negative_images)
            except Exception:
                print("Encountered exception in forward pass. batch_tensor has shape", batch_tensor.shape)
                break
            
            with torch.no_grad():
                loss = facenet.triplet_loss(anchor_embedding, positive_embedding, negative_embedding, alpha)
                _, _, accuracy = facenet.calculate_accuracy(anchor_embedding, positive_embedding, negative_embedding, alpha)

            loss_meter.update(loss, batch_size)
            accuracy_meter.update(accuracy, batch_size)

            batch_end = datetime.now()
            
            batch_str = "\tEvaluation {:d}/{:d} ({:d} triplets) finished in {:s}. Loss: {:.3f}. Accuracy: {:.3f}\n".format(batch_number, epoch_size, batch_size,
                str(batch_end-batch_start), float(loss_meter.val), float(accuracy_meter.val))
            print(batch_str[:-1])
            with open(log_file, "a") as f:
                f.write(batch_str)
    return loss_meter.val, accuracy_meter.val

"""
def train(args, sess, dataset, epoch, image_paths_placeholder, labels_placeholder, labels_batch,
          batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op, input_queue, global_step, 
          embeddings, loss, train_op, summary_op, summary_writer, learning_rate_schedule_file,
          embedding_size, anchor, positive, negative, triplet_loss):
    batch_number = 0
    
    if args.learning_rate>0.0:
        lr = args.learning_rate
    else:
        lr = facenet.get_learning_rate_from_file(learning_rate_schedule_file, epoch)
    while batch_number < args.epoch_size:
        # Sample people randomly from the dataset
        image_paths, num_per_class = sample_people(dataset, args.people_per_batch, args.images_per_person)
        
        print('Running forward pass on sampled images: ', end='')
        start_time = time.time()
        nrof_examples = args.people_per_batch * args.images_per_person
        labels_array = np.reshape(np.arange(nrof_examples),(-1,3))
        image_paths_array = np.reshape(np.expand_dims(np.array(image_paths),1), (-1,3))
        sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})
        emb_array = np.zeros((nrof_examples, embedding_size))
        nrof_batches = int(np.ceil(nrof_examples / args.batch_size))
        for i in range(nrof_batches):
            batch_size = min(nrof_examples-i*args.batch_size, args.batch_size)
            emb, lab = sess.run([embeddings, labels_batch], feed_dict={batch_size_placeholder: batch_size, 
                learning_rate_placeholder: lr, phase_train_placeholder: True})
            emb_array[lab,:] = emb
        print('%.3f' % (time.time()-start_time))

        # Select triplets based on the embeddings
        print('Selecting suitable triplets for training')
        triplets, nrof_random_negs, nrof_triplets = select_triplets(emb_array, num_per_class, 
            image_paths, args.people_per_batch, args.alpha)
        selection_time = time.time() - start_time
        print('(nrof_random_negs, nrof_triplets) = (%d, %d): time=%.3f seconds' % 
            (nrof_random_negs, nrof_triplets, selection_time))

        # Perform training on the selected triplets
        nrof_batches = int(np.ceil(nrof_triplets*3/args.batch_size))
        triplet_paths = list(itertools.chain(*triplets))
        labels_array = np.reshape(np.arange(len(triplet_paths)),(-1,3))
        triplet_paths_array = np.reshape(np.expand_dims(np.array(triplet_paths),1), (-1,3))
        sess.run(enqueue_op, {image_paths_placeholder: triplet_paths_array, labels_placeholder: labels_array})
        nrof_examples = len(triplet_paths)
        train_time = 0
        i = 0
        emb_array = np.zeros((nrof_examples, embedding_size))
        loss_array = np.zeros((nrof_triplets,))
        summary = tf.Summary()
        step = 0
        while i < nrof_batches:
            start_time = time.time()
            batch_size = min(nrof_examples-i*args.batch_size, args.batch_size)
            feed_dict = {batch_size_placeholder: batch_size, learning_rate_placeholder: lr, phase_train_placeholder: True}
            err, _, step, emb, lab = sess.run([loss, train_op, global_step, embeddings, labels_batch], feed_dict=feed_dict)
            emb_array[lab,:] = emb
            loss_array[i] = err
            duration = time.time() - start_time
            print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f' %
                  (epoch, batch_number+1, args.epoch_size, duration, err))
            batch_number += 1
            i += 1
            train_time += duration
            summary.value.add(tag='loss', simple_value=err)
            
        # Add validation loss and accuracy to summary
        #pylint: disable=maybe-no-member
        summary.value.add(tag='time/selection', simple_value=selection_time)
        summary_writer.add_summary(summary, step)
    return step
"""
def select_triplets(embeddings, nrof_images_per_class, image_paths, people_per_batch, alpha):
    """ Select the triplets for training
    """
    trip_idx = 0
    emb_start_idx = 0
    num_trips = 0
    triplets = []
    
    # VGG Face: Choosing good triplets is crucial and should strike a balance between
    #  selecting informative (i.e. challenging) examples and swamping training with examples that
    #  are too hard. This is achieve by extending each pair (a, p) to a triplet (a, p, n) by sampling
    #  the image n at random, but only between the ones that violate the triplet loss margin. The
    #  latter is a form of hard-negative mining, but it is not as aggressive (and much cheaper) than
    #  choosing the maximally violating example, as often done in structured output learning.

    for i in xrange(people_per_batch):
        nrof_images = int(nrof_images_per_class[i])
        for j in xrange(1,nrof_images):
            a_idx = emb_start_idx + j - 1
            neg_dists_sqr = np.sum(np.square(embeddings[a_idx] - embeddings), 1)
            for pair in xrange(j, nrof_images): # For every possible positive pair.
                p_idx = emb_start_idx + pair
                pos_dist_sqr = np.sum(np.square(embeddings[a_idx]-embeddings[p_idx]))
                neg_dists_sqr[emb_start_idx:emb_start_idx+nrof_images] = np.NaN
                #all_neg = np.where(np.logical_and(neg_dists_sqr-pos_dist_sqr<alpha, pos_dist_sqr<neg_dists_sqr))[0]  # FaceNet selection
                all_neg = np.where(neg_dists_sqr-pos_dist_sqr<alpha)[0] # VGG Face selecction
                nrof_random_negs = all_neg.shape[0]
                if nrof_random_negs>0:
                    rnd_idx = np.random.randint(nrof_random_negs)
                    n_idx = all_neg[rnd_idx]
                    triplets.append((image_paths[a_idx], image_paths[p_idx], image_paths[n_idx]))
                    #print('Triplet %d: (%d, %d, %d), pos_dist=%2.6f, neg_dist=%2.6f (%d, %d, %d, %d, %d)' % 
                    #    (trip_idx, a_idx, p_idx, n_idx, pos_dist_sqr, neg_dists_sqr[n_idx], nrof_random_negs, rnd_idx, i, j, emb_start_idx))
                    trip_idx += 1

                num_trips += 1

        emb_start_idx += nrof_images

    np.random.shuffle(triplets)
    return triplets, num_trips, len(triplets)

def sample_people(dataset, people_per_batch, images_per_person):
    nrof_images = people_per_batch * images_per_person
  
    # Sample classes from the dataset
    nrof_classes = len(dataset)
    class_indices = np.arange(nrof_classes)
    np.random.shuffle(class_indices)
    
    i = 0
    image_paths = []
    num_per_class = []
    sampled_class_indices = []
    # Sample images from these classes until we have enough
    while len(image_paths)<nrof_images:
        class_index = class_indices[i]
        nrof_images_in_class = len(dataset[class_index])
        image_indices = np.arange(nrof_images_in_class)
        np.random.shuffle(image_indices)
        nrof_images_from_class = min(nrof_images_in_class, images_per_person, nrof_images-len(image_paths))
        idx = image_indices[0:nrof_images_from_class]
        image_paths_for_class = [dataset[class_index].image_paths[j] for j in idx]
        sampled_class_indices += [class_index]*nrof_images_from_class
        image_paths += image_paths_for_class
        num_per_class.append(nrof_images_from_class)
        i+=1
  
    return image_paths, num_per_class
"""
def evaluate(sess, image_paths, embeddings, labels_batch, image_paths_placeholder, labels_placeholder, 
        batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op, actual_issame, batch_size, 
        nrof_folds, log_dir, step, summary_writer, embedding_size):
    start_time = time.time()
    # Run forward pass to calculate embeddings
    print('Running forward pass on LFW images: ', end='')
    
    nrof_images = len(actual_issame)*2
    assert(len(image_paths)==nrof_images)
    labels_array = np.reshape(np.arange(nrof_images),(-1,3))
    image_paths_array = np.reshape(np.expand_dims(np.array(image_paths),1), (-1,3))
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})
    emb_array = np.zeros((nrof_images, embedding_size))
    nrof_batches = int(np.ceil(nrof_images / batch_size))
    label_check_array = np.zeros((nrof_images,))
    for i in xrange(nrof_batches):
        batch_size = min(nrof_images-i*batch_size, batch_size)
        emb, lab = sess.run([embeddings, labels_batch], feed_dict={batch_size_placeholder: batch_size,
            learning_rate_placeholder: 0.0, phase_train_placeholder: False})
        emb_array[lab,:] = emb
        label_check_array[lab] = 1
    print('%.3f' % (time.time()-start_time))
    
    assert(np.all(label_check_array==1))
    
    _, _, accuracy, val, val_std, far = lfw.evaluate(emb_array, actual_issame, nrof_folds=nrof_folds)
    
    print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    lfw_time = time.time() - start_time
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='lfw/accuracy', simple_value=np.mean(accuracy))
    summary.value.add(tag='lfw/val_rate', simple_value=val)
    summary.value.add(tag='time/lfw', simple_value=lfw_time)
    summary_writer.add_summary(summary, step)
    with open(os.path.join(log_dir,'lfw_result.txt'),'at') as f:
        f.write('%d\t%.5f\t%.5f\n' % (step, np.mean(accuracy), val))
"""
def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0  
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)
  
  
def get_learning_rate_from_file(filename, epoch):
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.split('#', 1)[0]
            if line:
                par = line.strip().split(':')
                e = int(par[0])
                lr = float(par[1])
                if e <= epoch:
                    learning_rate = lr
                else:
                    return learning_rate
    

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--logs_base_dir', type=str, 
        help='Directory where to write event logs.', default='../logs/facenet')
    parser.add_argument('--models_base_dir', type=str,
        help='Directory where to write trained models and checkpoints.', default='../models/facenet')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--pretrained_model', type=str,
        help='Load a pretrained model before training starts.')
    parser.add_argument('--data_dir', type=str,
        help='Path to the data directory containing aligned face patches.',
        default='/mnt/usb/data/ravit/datasets/CASIA-WebFace')
    parser.add_argument('--model_def', type=str,
        help='Model definition. Points to a module containing the definition of the inference graph.', default='models.inception_resnet_v1')
    parser.add_argument('--max_nrof_epochs', type=int,
        help='Number of epochs to run.', default=25)
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--people_per_batch', type=int,
        help='Number of people per batch.', default=45)
    parser.add_argument('--images_per_person', type=int,
        help='Number of images per person.', default=40)
    parser.add_argument('--epoch_size', type=int,
        help='Number of batches per epoch.', default=1000)
    parser.add_argument('--alpha', type=float,
        help='Positive to negative triplet distance margin.', default=0.2)
    parser.add_argument('--embedding_size', type=int,
        help='Dimensionality of the embedding.', default=128)
    parser.add_argument('--random_crop', 
        help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
         'If the size of the images in the data directory is equal to image_size no cropping is performed', action='store_true')
    parser.add_argument('--random_flip', 
        help='Performs random horizontal flipping of training images.', action='store_true')
    parser.add_argument('--keep_probability', type=float,
        help='Keep probability of dropout for the fully connected layer(s).', default=1.0)
    parser.add_argument('--weight_decay', type=float,
        help='L2 weight regularization.', default=0.0)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
        help='The optimization algorithm to use', default='ADAGRAD')
    parser.add_argument('--learning_rate', type=float,
        help='Initial learning rate. If set to a negative value a learning rate ' +
        'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.1)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
        help='Number of epochs between learning rate decay.', default=100)
    parser.add_argument('--learning_rate_decay_factor', type=float,
        help='Learning rate decay factor.', default=1.0)
    parser.add_argument('--moving_average_decay', type=float,
        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--learning_rate_schedule_file', type=str,
        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.', default='data/learning_rate_schedule.txt')

    # Parameters for validation on LFW
    parser.add_argument('--lfw_pairs', type=str,
        help='The file containing the pairs to use for validation.', default='data/pairs.txt')
    parser.add_argument('--lfw_dir', type=str,
        help='Path to the data directory containing aligned face patches.', default='/mnt/usb/data/ravit/datasets/lfw')
    parser.add_argument('--lfw_nrof_folds', type=int,
        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    return parser.parse_args(argv)
  

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

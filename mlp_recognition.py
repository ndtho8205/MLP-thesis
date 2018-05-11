"""MLP Recognition model.
"""
import os

import numpy as np
import tensorflow as tf

from utils import load_model


class MLPRecognition:
    """MLPRecognition
    """
    MODEL_CHECKPOINT_NAME = 'thesis_mlp'
    N_CLASSES = 2
    LEARNING_RATE = 0.01
    TRAINING_EPOCHS = 500
    BATCH_SIZE = 100
    DISPLAY_STEP = 1

    # Network Parameters
    N_INPUT = 512
    N_HIDDEN_1 = 64
    N_HIDDEN_2 = 32

    def __init__(self, model_path):
        self.sess = None
        self.model_path = model_path

    def load(self):
        """Load the recognition model."""
        if self.model_path is not None:
            with tf.Graph().as_default():
                self.sess = tf.Session()
                self._neural_networks_construction()
                ckpt = tf.train.get_checkpoint_state(self.model_path)
                loader = tf.train.Saver()
                loader.restore(self.sess, ckpt.model_checkpoint_path)

    def fit(self, X_path, y_path, X_val_path=None, y_val_path=None):
        """Fit the recognition model according to the given training data."""
        print('Training data loaded')
        # load training data
        X_train = load_model.load(X_path)
        y_train = load_model.load(y_path)
        X_val = y_val = None
        if X_val_path is not None and y_val_path is not None:
            X_val = load_model.load(X_val_path)
            y_val = load_model.load(y_val_path)

        print('Training data loaded')

        with tf.Graph().as_default():
            self.sess = tf.Session()

            # construct neural networks
            _, loss, optimizer, accuracy = self._neural_networks_construction()
            print("Neural Networks built successfully")

            # saver
            saver = tf.train.Saver()

            # training
            print('Starting training...')
            self._train(X_train, y_train, loss, optimizer, accuracy, X_val,
                        y_val)
            saver.save(self.sess, os.path.join(self.model_path, './mlp_2'))
            print('End of training')

    def _train(self, X, y, loss, optimizer, accuracy, X_val=None, y_val=None):
        y = np.asarray(y, dtype=np.int64)
        with self.sess.as_default():
            self.sess.run(tf.global_variables_initializer())

            # training loop
            for epoch in range(self.TRAINING_EPOCHS):
                avg_loss = 0.
                train_accuracy = 0.
                total_batch = 1 if X.shape[0] < self.BATCH_SIZE else int(
                    X.shape[0] / self.BATCH_SIZE)

                # loop over all batches
                for i in range(total_batch):
                    start_index = i * self.BATCH_SIZE
                    end_index = min((i + 1) * self.BATCH_SIZE, len(y))
                    batch_y = y[start_index:end_index]
                    batch_X = X[start_index:end_index]

                    # fit using batched data
                    _, train_loss, train_accuracy = self.sess.run(
                        [optimizer, loss, accuracy],
                        feed_dict={
                            'X:0': batch_X,
                            'y:0': batch_y
                        })
                    # compute average loss
                    avg_loss += train_loss / total_batch

                # display progress
                if epoch % self.DISPLAY_STEP == 0:
                    print('Epoch: {}\tLoss: {:.9f}\tTraining accuracy: {:.3f}'.
                          format(epoch + 1, avg_loss, train_accuracy))
                    if X_val is not None and y_val is not None:
                        validate_loss, validate_accuracy = self.sess.run(
                            [loss, accuracy],
                            feed_dict={
                                'X:0': X_val,
                                'y:0': y_val,
                            })
                        print(
                            'Validate: loss: {:.3f}\taccuracy: {:.3f}'.format(
                                validate_loss, validate_accuracy))

    def _neural_networks_construction(self):
        # learning variables
        n_input = self.N_INPUT
        n_hidden_1 = self.N_HIDDEN_1
        # n_hidden_2 = self.N_HIDDEN_2
        n_classes = self.N_CLASSES

        X = tf.placeholder(tf.float32, [None, n_input], name='X')
        y = tf.placeholder(tf.int64, [None], name='y')

        weights = {
            'h1':
            tf.Variable(tf.random_normal([n_input, n_hidden_1]), name='w_h1'),
            'out':
            tf.Variable(
                tf.random_normal([n_hidden_1, n_classes]), name='w_out')
        }

        biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1]), name='b_b1'),
            'out': tf.Variable(tf.random_normal([n_classes]), name='b_out')
        }

        # construct model
        model = self._mlp(X, weights, biases)

        # loss and optimizer
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=model, labels=y),
            name='loss')
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.LEARNING_RATE).minimize(loss)

        # accuracy
        pred = tf.argmax(tf.nn.softmax(model), 1, name='pred')
        correct_prediction = tf.equal(pred, y, name='correct_prediction')
        accuracy = tf.reduce_mean(
            tf.cast(correct_prediction, tf.float32), name='accuracy')

        return model, loss, optimizer, accuracy

    def _mlp(self, X, weights, biases):
        layer_1 = tf.nn.relu(tf.add(tf.matmul(X, weights['h1']), biases['b1']))
        # layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
        out = tf.matmul(layer_1, weights['out']) + biases['out']
        return out

    def predict(self, X):
        """Perform recognition on samples in X."""
        X = np.asarray(X, dtype=np.float32)
        labels_index = self.sess.run('pred:0', {'X:0': X})
        print('Predict: {}'.format(labels_index))
        return labels_index[0]

    def predict_proba(self, X):
        """Compute probabilities of possible outcomes for samples in X."""
        X = np.asarray(X, dtype=np.float32)
        labels_index = self.sess.run('pred:0', {'X:0': X})
        print('Predict: {}'.format(labels_index))
        return labels_index[0], 0.

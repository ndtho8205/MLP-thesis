"""MLP model.
"""
import os
import shutil
import logging

import numpy as np
import tensorflow as tf

from utils import load_model


class MLP:
    """MLP
    """
    SAVED_MODEL_PATH = './models/mlp/'
    MODEL_NAME = 'mlpThesis'
    N_CLASSES = 2
    LEARNING_RATE = 0.01
    TRAINING_EPOCHS = 500
    BATCH_SIZE = 100
    DISPLAY_STEP = BATCH_SIZE // 10

    # Network Parameters
    N_INPUT = 512
    N_HIDDEN_1 = 64
    N_HIDDEN_2 = 32

    def __init__(self):
        super().__init__()
        self.sess = None

        self.log_path = './logs'
        self.train_log_path = './logs/train'
        self.validate_log_path = './logs/validate'

        if os.path.exists(self.log_path):
            shutil.rmtree(self.log_path)
            os.makedirs(self.log_path)

    def load(self):
        """Load the model."""
        if self.SAVED_MODEL_PATH is not None:
            with tf.Graph().as_default():
                self.sess = tf.Session()
                self._neural_networks_construction()
                ckpt = tf.train.get_checkpoint_state(self.SAVED_MODEL_PATH)
                loader = tf.train.Saver()
                loader.restore(self.sess, ckpt.model_checkpoint_path)

    def fit(self, X_path, y_path, X_val_path=None, y_val_path=None):
        """Fit the model according to the given training data."""
        logging.info('Training data loaded')
        # load training data
        X_train = load_model.load(X_path)
        y_train = load_model.load(y_path)
        X_val = y_val = None
        if X_val_path is not None and y_val_path is not None:
            X_val = load_model.load(X_val_path)
            y_val = load_model.load(y_val_path)

        logging.info('Training data loaded')

        with tf.Graph().as_default():
            self.sess = tf.Session()

            # construct neural networks
            _, loss, optimizer, accuracy = self._neural_networks_construction()
            logging.info("Neural Networks built successfully")

            # saver
            saver = tf.train.Saver()

            # training
            logging.info('Starting training...')
            self._train(X_train, y_train, loss, optimizer, accuracy, X_val, y_val)
            saver.save(self.sess, self.SAVED_MODEL_PATH + self.MODEL_NAME + '.ckpt')

            logging.info('Saving model...')
            self.write_graph()

            logging.info('End of training')

    def write_graph(self):
        tf.train.write_graph(self.sess.graph_def, self.SAVED_MODEL_PATH, self.MODEL_NAME + '.pbtxt')
        tf.train.write_graph(self.sess.graph_def, self.SAVED_MODEL_PATH, self.MODEL_NAME + '.pb', as_text=False)

        self.freeze_graph()

    def freeze_graph(self):
        from tensorflow.python.tools import freeze_graph
        # Freeze the graph
        input_graph = self.SAVED_MODEL_PATH + self.MODEL_NAME + '.pb'
        input_saver = ""
        input_binary = True
        input_checkpoint = self.SAVED_MODEL_PATH + self.MODEL_NAME + '.ckpt'
        output_node_names = 'y_pred'
        restore_op_name = 'save/restore_all'
        filename_tensor_name = 'save/Const:0'
        output_graph = self.SAVED_MODEL_PATH + 'frozen_' + self.MODEL_NAME + '.pb'
        clear_devices = True
        initializer_nodes = ""
        variable_names_blacklist = ""

        freeze_graph.freeze_graph(input_graph, input_saver, input_binary, input_checkpoint, output_node_names,
                                  restore_op_name, filename_tensor_name, output_graph, clear_devices, initializer_nodes,
                                  variable_names_blacklist)

    def _train(self, X, y, loss, optimizer, accuracy, X_val=None, y_val=None):
        y = np.asarray(y, dtype=np.int64)

        with self.sess.as_default():
            # writer
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(self.train_log_path, self.sess.graph)
            validate_writer = tf.summary.FileWriter(self.validate_log_path)

            self.sess.run(tf.global_variables_initializer())

            step = 0

            # training loop
            for epoch in range(self.TRAINING_EPOCHS):
                avg_loss = 0.
                train_accuracy = 0.
                total_batch = 1 if X.shape[0] < self.BATCH_SIZE else int(X.shape[0] / self.BATCH_SIZE)

                # loop over all batches
                for i in range(total_batch):
                    start_index = i * self.BATCH_SIZE
                    end_index = min((i + 1) * self.BATCH_SIZE, len(y))
                    batch_y = y[start_index:end_index]
                    batch_X = X[start_index:end_index]

                    # fit using batched data
                    train_summary, _, train_loss, train_accuracy = self.sess.run(
                        [merged, optimizer, loss, accuracy], feed_dict={
                            'X:0': batch_X,
                            'y:0': batch_y
                        })
                    # compute average loss
                    avg_loss += train_loss / total_batch

                    # save summary
                    if step % self.DISPLAY_STEP == 0:
                        train_writer.add_summary(train_summary, step)
                        print('Train: {:.9f} - {:.3f}'.format(train_loss, train_accuracy))
                        if X_val is not None and y_val is not None:
                            validate_summary, validate_loss, validate_accuracy = self.sess.run(
                                [merged, loss, accuracy], feed_dict={
                                    'X:0': X_val,
                                    'y:0': y_val,
                                })
                            validate_writer.add_summary(validate_summary, step)
                            print('Validate: {:.9f} - {:.3f}'.format(validate_loss, validate_accuracy))
                    step += 1

                    # display progress
                    # if epoch % self.DISPLAY_STEP == 0:
                    # print('Epoch: {}\tLoss: {:.9f}\tTraining accuracy: {:.3f}'.format(
                    # epoch + 1, avg_loss, train_accuracy))
                    # if X_val is not None and y_val is not None:
                    #    validate_loss, validate_accuracy = self.sess.run(
                    #        [loss, accuracy], feed_dict={
                    #            'X:0': X_val,
                    #            'y:0': y_val,
                    #        })
                    # print('Validate: loss: {:.3f}\taccuracy: {:.3f}'.format(validate_loss, validate_accuracy))

            train_writer.close()
            validate_writer.close()

    def _neural_networks_construction(self):
        # learning variables
        n_input = self.N_INPUT
        n_hidden_1 = self.N_HIDDEN_1
        # n_hidden_2 = self.N_HIDDEN_2
        n_classes = self.N_CLASSES

        X = tf.placeholder(tf.float32, [None, n_input], name='X')
        y = tf.placeholder(tf.int64, [None], name='y')

        weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]), name='w_h1'),
            'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]), name='w_out')
        }
        # self._variable_summaries(weights['h1'])
        # self._variable_summaries(weights['out'])

        biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1]), name='b_b1'),
            'out': tf.Variable(tf.random_normal([n_classes]), name='b_out')
        }

        # construct model
        model = self._mlp(X, weights, biases)

        # loss and optimizer
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=y), name='loss')
        tf.summary.scalar("loss_function", loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.LEARNING_RATE).minimize(loss)

        # accuracy
        pred = tf.argmax(tf.nn.softmax(model), 1, name='y_pred')
        correct_prediction = tf.equal(pred, y, name='correct_prediction')
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
        tf.summary.scalar("accuracy_function", accuracy)

        return model, loss, optimizer, accuracy

    def _mlp(self, X, weights, biases):
        layer_1 = tf.nn.relu(tf.add(tf.matmul(X, weights['h1']), biases['b1']))
        # layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
        out = tf.matmul(layer_1, weights['out']) + biases['out']
        return out

    def _variable_summaries(self, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def predict(self, X):
        """Perform recognition on samples in X."""
        X = np.asarray(X, dtype=np.float32)
        labels_index = self.sess.run('y_pred:0', {'X:0': X})
        logging.info('Predict: {}'.format(labels_index))
        return labels_index[0]

    def predict_proba(self, X):
        """Compute probabilities of possible outcomes for samples in X."""
        X = np.asarray(X, dtype=np.float32)
        labels_index = self.sess.run('y_pred:0', {'X:0': X})
        logging.info('Predict: {}'.format(labels_index))
        return labels_index[0], 0.

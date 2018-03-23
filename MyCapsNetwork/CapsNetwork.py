import numpy as np
import tensorflow as tf
from MyCapsNetwork.CapsDecoder import *


class CapsNetwork(object):
    # primary capsules
    caps1_maps = 32
    caps1_caps = caps1_maps * 6 * 6  # 1152 primary capsules
    caps1_vec_length = 8

    # digit capsules
    caps2_caps = 10
    caps2_vec_length = 16
    init_sigma = 0.1

    # margin loss
    m_plus = 0.9
    m_minus = 0.1
    lambda_ = 0.5

    # final loss
    alpha = 0.0005

    # other
    routing_by_agreement_iterations = 2

    def __init__(self, X: tf.Tensor, X_raw: tf.Tensor, checkpoint_path: str):
        self._X_raw = X_raw
        self.checkpoint_path = checkpoint_path

        self.caps_outputs = []

        caps1_output = self.squash(X, name="caps1_output")
        W_init = tf.random_normal(shape=(1, self.caps1_caps, self.caps2_caps, self.caps2_vec_length, self.caps1_vec_length),
            stddev=self.init_sigma, dtype=tf.float32, name="W_init")
        W = tf.Variable(W_init, name="W")
        self.batch_size = tf.shape(caps1_output)[0]
        self.W_tiled = tf.tile(W, [self.batch_size, 1, 1, 1, 1], name="W_tiled")

        caps1_output_expanded = tf.expand_dims(caps1_output, -1, name="caps1_output_expanded")
        caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2, name="caps1_output_tile")
        caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, self.caps2_caps, 1, 1], name="caps1_output_tiled")

        self.y = tf.placeholder(shape=[None], dtype=tf.int64, name="y")

        self.caps2_predicted = tf.matmul(self.W_tiled, caps1_output_tiled, name="caps2_predicted")
        self.caps2_output = self.__routing_by_agreement()
        self.y_pred = self.__predict()
        self.margin_loss = self.__calc_margin_loss()
        self.decoder = CapsDecoder(self)
        self.loss = self.__calc_loss()
        self.accuracy = self.__calc_accuracy()
        self.training_optimizer = self.__build_optimizer()

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

    def caps_layer(self):
        caps_output_tield = self.__tile_output(self.caps_outputs[-1])
        caps_predicted = tf.matmul(self.W_tiled, caps_output_tield)

    def __tile_output(self, caps_output):
        caps_output_expanded = tf.expand_dims(caps_output, -1, name="caps1_output_expanded")
        caps_output_tile = tf.expand_dims(caps_output_expanded, 2, name="caps1_output_tile")
        return tf.tile(caps_output_tile, [1, 1, self.caps2_caps, 1, 1], name="caps1_output_tiled")

    def apply_options(self, options: {float}):
        # primary capsules
        self.caps1_maps = self.get_or_default(options, 'caps1_maps', 32)
        self.caps1_caps = self.get_or_default(options, 'caps1_caps', self.caps1_maps * 6 * 6)
        self.caps1_vec_length = self.get_or_default(options, 'caps1_vec_length', 8)

        # digit capsules
        self.caps2_caps = self.get_or_default(options, 'caps2_caps', 10)
        self.caps2_vec_length = self.get_or_default(options, 'caps2_vec_length', 16)

        self.init_sigma = self.get_or_default(options, 'init_sigma', 0.1)

        # margin loss
        self.m_plus = self.get_or_default(options, 'm_plus', 0.9)
        self.m_minus = self.get_or_default(options, 'm_minus', 0.1)
        self.lambda_ = self.get_or_default(options, 'lambda_', 0.5)

        # final loss
        self.alpha = self.get_or_default(options, 'alpha', 0.005)

    @staticmethod
    def get_or_default(options: {float}, key: str, default):
        if key in options:
            return options[key]
        else:
            return default

    def __routing_by_agreement(self):
        # TODO make a loop

        with tf.name_scope('routing_by_agreement'):
            # 2: b_ij
            raw_weigts = tf.zeros([self.batch_size, self.caps1_caps, self.caps2_caps, 1, 1], dtype=np.float32,
                                  name="raw_weights")
            # 4: c_i = softmax(b_i)
            routing_weights = tf.nn.softmax(raw_weigts, dim=2, name="routing_weights")
            # 5: sum c_ij * û_j|i
            weighted_predictions = tf.multiply(routing_weights, self.caps2_predicted, name="weighted_predictions")
            weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keep_dims=True, name="weigthed_sum")
            # 6: squash(sj)
            caps2_output_round_1 = self.squash(weighted_sum, axis=2, name="caps2_output_round_1")

            # 7: agreement = û_j|i dot v_j
            caps2_output_round_1_tiled = tf.tile(caps2_output_round_1, [1, self.caps1_caps, 1, 1, 1],
                                                 name="caps2_output_round_1_tiled")
            agreement = tf.matmul(self.caps2_predicted, caps2_output_round_1_tiled, transpose_a=True, name="agreement")
            # 7: b_j|i = b_ij + agreement
            raw_weights_round_2 = tf.add(raw_weigts, agreement, name="raw_weights_round_2")

            routing_weights_round_2 = tf.nn.softmax(raw_weights_round_2, dim=2, name="routing_weights_round_2")
            weighted_predictions_round_2 = tf.multiply(routing_weights_round_2, self.caps2_predicted,
                                                       name="weighted_predictions_round_2")
            weighted_sum_round_2 = tf.reduce_sum(weighted_predictions_round_2, axis=1, keep_dims=True,
                                                 name="weighted_sum_round_2")
            caps2_output_round_2 = self.squash(weighted_sum_round_2, axis=-2, name="caps2_output_round_2")

            return caps2_output_round_2

    def __routing_by_agreement2(self):
            #def condition(input, counter):
            #    return tf.less(counter, self.routing_by_agreement_iterations)
        
            #def loop_body(input, counter):
            #    # 2: b_ij
            #    raw_weigts = tf.zeros([self.batch_size, prev_caps, current_caps, 1, 1], dtype=np.float32,
            #                        name="raw_weights")
            #    # 4: c_i = softmax(b_i)
            #    routing_weights = tf.nn.softmax(raw_weigts, dim=2, name="routing_weights")
            #    # 5: sum c_ij * û_j|i
            #    weighted_predictions = tf.multiply(routing_weights, self.caps2_predicted, name="weighted_predictions")
            #    weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keep_dims=True, name="weigthed_sum")
            #    # 6: squash(sj)
            #    caps_output = self.squash(weighted_sum, axis=2, name="caps2_output_round_1")
        
            #    # 7: agreement = û_j|i dot v_j
            #    caps_output_tiled = tf.tile(caps_output, [1, self.caps1_caps, 1, 1, 1],
            #                                        name="caps2_output_round_1_tiled")
            #    agreement = tf.matmul(self.caps2_predicted, caps_output_tiled, transpose_a=True, name="agreement")
            #    # 7: b_j|i = b_ij + agreement
            #    raw_weights_round_2 = tf.add(raw_weigts, agreement, name="raw_weights_round_2")
        
            #    return caps_output, tf.add(counter, 1)

        prev_caps = self.caps1_caps
        current_caps = self.caps2_caps

        # 2: b_ij
        raw_weigts = tf.zeros([self.batch_size, prev_caps, current_caps, 1, 1], dtype=np.float32,
                            name="raw_weights")
        
        caps_output = tf.zeros([self.batch_size, prev_caps, current_caps, 1, 1], dtype=np.float32, name="caps2_output")

        for i in range(0, 4):
            # 4: c_i = softmax(b_i)
            routing_weights = tf.nn.softmax(raw_weigts, dim=2, name="routing_weights")
            # 5: sum c_ij * û_j|i
            weighted_predictions = tf.multiply(routing_weights, self.caps2_predicted, name="weighted_predictions")
            weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keep_dims=True, name="weigthed_sum")
            # 6: squash(sj)
            caps_output = self.squash(weighted_sum, axis=2, name="caps2_output")
        
            # 7: agreement = û_j|i dot v_j
            caps_output_tiled = tf.tile(caps_output, [1, self.caps1_caps, 1, 1, 1],
                                                name="caps2_output_round_1_tiled")
            agreement = tf.matmul(self.caps2_predicted, caps_output_tiled, transpose_a=True, name="agreement")
            # 7: b_j|i = b_ij + agreement
            raw_weights_round = tf.add(raw_weigts, agreement, name="raw_weights_round_2")

        return caps_output
   
       

    #i = tf.constant(0)
    #c = lambda i: tf.less(i, 10)
    #r = tf.while_loop(c, cond, [i])

    #    with tf.name_scope("compute_sum_of_squares"):
    #        counter = tf.constant(1)
    #        sum_of_squares = tf.constant(0)
        
    #        result = tf.while_loop(condition, loop_body, [sum_of_squares, counter])

    def loop_body(i):
        # 2: b_ij
        raw_weigts = tf.zeros([self.batch_size, prev_caps, current_caps, 1, 1], dtype=np.float32,
                            name="raw_weights")
        # 4: c_i = softmax(b_i)
        routing_weights = tf.nn.softmax(raw_weigts, dim=2, name="routing_weights")
        # 5: sum c_ij * û_j|i
        weighted_predictions = tf.multiply(routing_weights, self.caps2_predicted, name="weighted_predictions")
        weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keep_dims=True, name="weigthed_sum")
        # 6: squash(sj)
        caps_output = self.squash(weighted_sum, axis=2, name="caps2_output_round_1")
        
        # 7: agreement = û_j|i dot v_j
        caps_output_tiled = tf.tile(caps_output, [1, self.caps1_caps, 1, 1, 1],
                                            name="caps2_output_round_1_tiled")
        agreement = tf.matmul(self.caps2_predicted, caps_output_tiled, transpose_a=True, name="agreement")
        # 7: b_j|i = b_ij + agreement
        raw_weights_round = tf.add(raw_weigts, agreement, name="raw_weights_round_2")
        
        return tf.add(i, 1)
    
    def __predict(self):
        with tf.name_scope('predict'):
            y_proba = self.safe_norm(self.caps2_output, axis=-2, name="y_proba")
            y_proba_argmax = tf.argmax(y_proba, axis=2, name="y_proba_argmax")
            return tf.squeeze(y_proba_argmax, axis=[1, 2], name="y_pred")

    def __calc_margin_loss(self):
        with tf.name_scope('calc_margin_loss'):
            T = tf.one_hot(self.y, depth=self.caps2_caps, name="T")
            caps2_output_norm = self.safe_norm(self.caps2_output, axis=-2, keep_dims=True, name="caps2_output_norm")
            # max(0, m_+ - ||v_k||) ** 2
            present_error_raw = tf.square(tf.maximum(0., self.m_plus - caps2_output_norm), name="present_error_raw")
            present_error = tf.reshape(present_error_raw, shape=(-1, 10), name="present_error")
            # max(0, ||v_k|| - m_-) ** 2
            absent_error_raw = tf.square(tf.maximum(0., caps2_output_norm - self.m_minus), name="absent_error_raw")
            absent_error = tf.reshape(absent_error_raw, shape=(-1, 10), name="absent_error")
            # L_k = T_k * present_error + λ (1 - T_k) * absent_error
            L = tf.add(T * present_error, self.lambda_ * (1.0 - T) * absent_error, name="L")
            return tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss")

    def __calc_loss(self):
        with tf.name_scope('calc_loss'):
            return tf.add(self.margin_loss, self.alpha * self.decoder.calc_reconstruction_loss(self._X_raw))

    def __calc_accuracy(self):
        with tf.name_scope('calc_accuracy'):
            correct = tf.equal(self.y, self.y_pred, name="correct")
            return tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

    def __build_optimizer(self):
        with tf.name_scope('build_optimizer'):
            optimizer = tf.train.AdamOptimizer()
            return optimizer.minimize(self.loss, name="training_op")

    def loop_example(self):
        def condition(input, counter):
            return tf.less(counter, 100)

        def loop_body(input, counter):
            output = tf.add(input, tf.square(counter))
            return output, tf.add(counter, 1)

        with tf.name_scope("compute_sum_of_squares"):
            counter = tf.constant(1)
            sum_of_squares = tf.constant(0)

            result = tf.while_loop(condition, loop_body, [sum_of_squares, counter])

        with tf.Session() as sess:
            print(sess.run(result))

    def train(self, mnist, epochs: int, batch_size: int=50, restore_checkpoint=True):
        n_iterations_per_epoch = mnist.train.num_examples // batch_size
        n_iterations_validation = mnist.validation.num_examples // batch_size
        best_loss_val = np.infty

        with tf.Session() as sess:
            if restore_checkpoint and tf.train.checkpoint_exists(self.checkpoint_path):
                self.saver.restore(sess, self.checkpoint_path)
            else:
                self.init.run()

            writer = tf.summary.FileWriter("./logs", sess.graph)

            for epoch in range(epochs):
                for iteration in range(1, n_iterations_per_epoch + 1):
                    X_batch, y_batch = mnist.train.next_batch(batch_size)
                    # Run the training operation and measure the loss:
                    _, loss_train = sess.run([self.training_optimizer, self.loss],
                        feed_dict={self._X_raw: X_batch.reshape([-1, 28, 28, 1]),
                                   self.y: y_batch,
                                   self.decoder.mask_with_labels: True})

                    tf.summary.scalar('loss', self.loss)

                    print("\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f}".format(iteration, n_iterations_per_epoch,
                        iteration * 100 / n_iterations_per_epoch,
                        loss_train),
                        end="")

                # At the end of each epoch,
                # measure the validation loss and accuracy:
                loss_vals = []
                acc_vals = []
                for iteration in range(1, n_iterations_validation + 1):
                    X_batch, y_batch = mnist.validation.next_batch(batch_size)
                    loss_val, acc_val = sess.run([self.loss, self.accuracy],
                        feed_dict={self._X_raw: X_batch.reshape([-1, 28, 28, 1]),
                                   self.y: y_batch})
                    loss_vals.append(loss_val)
                    acc_vals.append(acc_val)
                    print("\rEvaluating the model: {}/{} ({:.1f}%)".format(iteration, n_iterations_validation,
                        iteration * 100 / n_iterations_validation),
                        end=" " * 10)
                loss_val = np.mean(loss_vals)
                acc_val = np.mean(acc_vals)
                print("\rEpoch: {}  Val accuracy: {:.4f}%  Loss: {:.6f}{}".format(epoch + 1, acc_val * 100, loss_val,
                    " (improved)" if loss_val < best_loss_val else ""))

                # And save the model if it improved:
                if loss_val < best_loss_val:
                    save_path = self.saver.save(sess, self.checkpoint_path)
                    best_loss_val = loss_val

            writer.close()

    def eval(self, mnist, batch_size=50):
        n_iterations_test = mnist.test.num_examples // batch_size

        with tf.Session() as sess:
            self.saver.restore(sess, self.checkpoint_path)

            loss_tests = []
            acc_tests = []
            for iteration in range(1, n_iterations_test + 1):
                X_batch, y_batch = mnist.test.next_batch(batch_size)
                loss_test, acc_test = sess.run([self.loss, self.accuracy],
                    feed_dict={self._X_raw: X_batch.reshape([-1, 28, 28, 1]),
                               self.y: y_batch})
                loss_tests.append(loss_test)
                acc_tests.append(acc_test)
                print("\rEvaluating the model: {}/{} ({:.1f}%)".format(iteration, n_iterations_test,
                    iteration * 100 / n_iterations_test),
                    end=" " * 10)
            loss_test = np.mean(loss_tests)
            acc_test = np.mean(acc_tests)
            print("\rFinal test accuracy: {:.4f}%  Loss: {:.6f}".format(acc_test * 100, loss_test))

    def predict_and_reconstruct(self, x):
        with tf.Session() as sess:
            self.saver.restore(sess, self.checkpoint_path)
            caps2_output_value, decoder_output_value, y_pred_value = sess.run([self.caps2_output, self.decoder.decoder_output, self.y_pred],
                feed_dict={self._X_raw: x,
                           self.y: np.array([], dtype=np.int64)})

        return caps2_output_value, decoder_output_value, y_pred_value

    @staticmethod
    def squash(s, axis=-1, epsilon=1e-7, name=None):
        with tf.name_scope(name, default_name="squash"):
            squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                         keep_dims=True)
            safe_norm = tf.sqrt(squared_norm + epsilon)
            squash_factor = squared_norm / (1. + squared_norm)
            unit_vector = s / safe_norm
            return squash_factor * unit_vector

    @staticmethod
    def safe_norm(s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
        with tf.name_scope(name, default_name="safe_norm"):
            squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keep_dims=keep_dims)
            return tf.sqrt(squared_norm + epsilon)

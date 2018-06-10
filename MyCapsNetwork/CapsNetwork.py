import numpy as np
import tensorflow as tf
from MyCapsNetwork.CapsDecoder import *            
import time


class CapsNetwork(object):
    # digit capsules
    caps2_caps = 10
    caps2_vec_norm = 16
    init_sigma = 0.1

    # margin loss
    m_plus = 0.9
    m_minus = 0.1
    lambda_ = 0.5

    # final loss
    alpha = 0.0005

    # other
    routing_by_agreement_iterations = 2

    def __init__(self, X: tf.Tensor, X_raw: tf.Tensor, checkpoint_path: str, caps1_caps=1152, caps1_vec_norm = 8, caps2_caps = 10, caps2_vec_dim = 16, decoder_output=784):

        self.stats = {}
        self.stats['train_acc'] = []
        self.stats['train_loss'] = []
        self.stats['val_acc'] = []
        self.stats['val_loss'] = []
        self.stats['time'] = []

        self.caps1_caps = caps1_caps
        self.caps1_vec_norm = caps1_vec_norm
        self.caps2_caps = caps2_caps
        self.caps2_vec_norm = caps2_vec_dim
        
        self._X_raw = X_raw
        self.checkpoint_path = checkpoint_path
        self.caps_outputs = []

        caps1_output = self.squash(X, name="caps1_output")
        self.batch_size = tf.shape(caps1_output)[0]
        self.W_tiled = self.__init_W()

        caps1_output_expanded = tf.expand_dims(caps1_output, -1, name="caps1_output_expanded")
        caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2, name="caps1_output_tile")
        caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, self.caps2_caps, 1, 1], name="caps1_output_tiled")

        self.y = tf.placeholder(shape=[None], dtype=tf.int64, name="y")

        self.mask_with_labels = tf.placeholder_with_default(False, shape=(), name="mask_with_labels")
        self.caps2_output_shifted = tf.placeholder_with_default(np.zeros(caps2_vec_dim, dtype=np.float32),shape=[caps2_vec_dim])

        self.caps2_predicted = tf.matmul(self.W_tiled, caps1_output_tiled, name="caps2_predicted")
        #tf.summary.tensor_summary("caps2_predicted", self.caps2_predicted)
        self.caps2_output = self.__routing_by_agreement()
        #tf.summary.tensor_summary("caps2_output", self.caps2_output)
        print(self.caps2_output)
        caps2_output_shifted_tiled = tf.reshape(self.caps2_output_shifted, (1,1,1,16,1))
        self.caps2_output = tf.add(self.caps2_output, caps2_output_shifted_tiled)
        self.y_pred = self.__predict()
        self.margin_loss = self.__calc_margin_loss()
        self.decoder = CapsDecoder(self, decoder_output)
        self.loss = self.__calc_loss()
        tf.summary.scalar('loss', self.loss)
        self.accuracy = self.__calc_accuracy()
        tf.summary.scalar('accuracy', self.accuracy)
        self.training_optimizer = self.__build_optimizer()

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

        self.merge = tf.summary.merge_all()

    def __init_W(self):
        with tf.name_scope('init_W'):
            W_init = tf.random_normal(shape=(1, self.caps1_caps, self.caps2_caps, self.caps2_vec_norm, self.caps1_vec_norm),
                stddev=self.init_sigma, dtype=tf.float32, name="W_init")
            W = tf.Variable(W_init, name="W")
            return tf.tile(W, [self.batch_size, 1, 1, 1, 1], name="W_tiled")

    def caps_layer(self):
        caps_output_tield = self.__tile_output(self.caps_outputs[-1])
        caps_predicted = tf.matmul(self.W_tiled, caps_output_tield)

    def __tile_output(self, caps_output):
        caps_output_expanded = tf.expand_dims(caps_output, -1, name="caps1_output_expanded")
        caps_output_tile = tf.expand_dims(caps_output_expanded, 2, name="caps1_output_tile")
        return tf.tile(caps_output_tile, [1, 1, self.caps2_caps, 1, 1], name="caps1_output_tiled")


    def __routing_by_agreement(self):
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
            present_error = tf.reshape(present_error_raw, shape=(-1, self.caps2_caps), name="present_error")
            # max(0, ||v_k|| - m_-) ** 2
            absent_error_raw = tf.square(tf.maximum(0., caps2_output_norm - self.m_minus), name="absent_error_raw")
            absent_error = tf.reshape(absent_error_raw, shape=(-1, self.caps2_caps), name="absent_error")
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

    def train(self, x_train: np.ndarray, y_train : np.ndarray, x_val : np.ndarray, y_val : np.ndarray, epochs: int, batch_size: int=100, restore_checkpoint=True):
        n_iterations_per_epoch = x_train.shape[0] // batch_size
        n_iterations_validation = x_val.shape[0] // batch_size
        best_loss_val = np.infty


        with tf.Session() as sess:
            if restore_checkpoint and tf.train.checkpoint_exists(self.checkpoint_path):
                self.saver.restore(sess, self.checkpoint_path)
            else:
                self.init.run()

            writer = tf.summary.FileWriter("./logs", sess.graph)

            start = time.time()

            for epoch in range(epochs):
                epoch_start = time.time()
                batch_index = 0
                loss_trains = []
                acc_trains = []
                for iteration in range(1, n_iterations_per_epoch + 1):
                    X_batch = x_train[batch_index:batch_index+batch_size-1]
                    y_batch = y_train[batch_index:batch_index+batch_size-1]

                    # Run the training operation and measure the loss:
                    _, loss_train, accuracy_train, summary = sess.run([self.training_optimizer, self.loss, self.accuracy, self.merge],
                        feed_dict={self._X_raw: X_batch,
                                   self.y: y_batch,
                                   self.decoder.mask_with_labels: True})

                    loss_trains.append(loss_train)
                    acc_trains.append(accuracy_train)

                    print("\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f} Accuracy: {:.5f}".format(iteration, n_iterations_per_epoch,
                        iteration * 100 / n_iterations_per_epoch,
                        loss_train, accuracy_train),
                        end="")
                    batch_index += batch_size
                writer.add_summary(summary, epoch)

                end = time.time()
                diff = end - epoch_start;
                print(f"Epoch time: {diff}");
                loss_train = np.mean(loss_trains)
                acc_train = np.mean(acc_trains)
                self.stats['train_acc'].append(acc_train)
                self.stats['train_loss'].append(loss_train)


                # At the end of each epoch,
                # measure the validation loss and accuracy:
                batch_index = 0
                loss_vals = []
                acc_vals = []
                for iteration in range(1, n_iterations_validation + 1):
                    X_batch = x_val[batch_index:batch_index+batch_size-1]
                    y_batch = y_val[batch_index:batch_index+batch_size-1]
                    loss_val, acc_val = sess.run([self.loss, self.accuracy],
                        feed_dict={self._X_raw: X_batch,
                                   self.y: y_batch})
                    loss_vals.append(loss_val)
                    acc_vals.append(acc_val)
                    print("\rEvaluating the model: {}/{} ({:.1f}%)".format(iteration, n_iterations_validation,
                        iteration * 100 / n_iterations_validation),
                        end=" " * 10)
                    batch_index += batch_size
                loss_val = np.mean(loss_vals)
                acc_val = np.mean(acc_vals)
                print("\rEpoch: {}  Val accuracy: {:.4f}%  Loss: {:.6f}{}".format(epoch + 1, acc_val * 100, loss_val,
                    " (improved)" if loss_val < best_loss_val else ""))

                self.stats['val_acc'].append(acc_val)
                self.stats['val_loss'].append(loss_val)

                # And save the model if it improved:
                if loss_val < best_loss_val:
                    save_path = self.saver.save(sess, self.checkpoint_path)
                    best_loss_val = loss_val
                
                diff2 = time.time() - start
                print(f"Total time: {diff2}")
                self.stats['time'].append(diff2)
                if diff2 > 60*10:
                    break;
                
            writer.close()

            print("train acc")
            print(self.stats['train_acc'])
            print()
            print("train loss")
            print(self.stats['train_loss'])
            print()
            print("val acc")
            print(self.stats['val_acc'])
            print()
            print("val loss")
            print(self.stats['val_loss'])
            print()
            print("time")
            print(self.stats['time'])
            print()

    def eval(self, x_test, y_test, batch_size=50):
        n_iterations_test = x_test.shape[0] // batch_size

        with tf.Session() as sess:
            self.saver.restore(sess, self.checkpoint_path)

            loss_tests = []
            acc_tests = []
            batch_index = 0
            for iteration in range(1, n_iterations_test + 1):
                X_batch =  x_test[batch_index:batch_index+batch_size-1]
                y_batch = y_test[batch_index:batch_index+batch_size-1]
                loss_test, acc_test = sess.run([self.loss, self.accuracy],
                    feed_dict={self._X_raw: X_batch,
                               self.y: y_batch})
                loss_tests.append(loss_test)
                acc_tests.append(acc_test)
                print("\rEvaluating the model: {}/{} ({:.1f}%)".format(iteration, n_iterations_test,
                    iteration * 100 / n_iterations_test),
                    end=" " * 10)
                batch_index += batch_size
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


    def predict(self, x):
        with tf.Session() as sess:
            self.saver.restore(sess, self.checkpoint_path)
            y_pred_value = sess.run([self.y_pred],
                feed_dict={self._X_raw: x,
                           self.y: np.array([], dtype=np.int64)})
            return y_pred_value


    #def batch_predict(self, x, batch_size=100):
    #    n_iterations = x.shape[0] // batch_size

    #    with tf.Session() as sess:
    #        self.saver.restore(sess, self.checkpoint_path)

    #        batch_index = 0
    #        for iteration in range(1, n_iterations + 1):
    #            X_batch =  x_test[batch_index:batch_index+batch_size-1]

    #            y_pred_value = sess.run([self.y_pred],
    #            feed_dict={self._X_raw: X_batch,
    #                       self.y: np.array([], dtype=np.int64)})

    #            print("\rPredict: {}/{} ({:.1f}%)".format(iteration, n_iterations_test,
    #                iteration * 100 / n_iterations_test),
    #                end=" " * 10)
    #            batch_index += batch_size


    def recunstruct_shifted(self, x, y, shift):
        with tf.Session() as sess:
            self.saver.restore(sess, self.checkpoint_path)
            caps2_output_value, manipulated_decoder_output_value, y_pred_value = sess.run([self.caps2_output, self.decoder.decoder_output, self.y_pred],
                feed_dict={self._X_raw: x,
                           self.y: y,
                           self.caps2_output_shifted: shift,
                           self.decoder.mask_with_labels: True})

        #print("caps2_output_value")
        #print(caps2_output_value.shape)
        #print(caps2_output_value[0,0,0])
        #print("Mainipulated")
        #print(caps2_output_value_manipulated.shape)
        #print(caps2_output_value_manipulated[0,0,0])

        return caps2_output_value, manipulated_decoder_output_value, y_pred_value

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

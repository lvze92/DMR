# coding: utf-8

from model import *
from data_iterator import *
import numpy as np
import tensorflow as tf
import sys
import random
from datetime import timedelta, datetime

num_epochs = 1
batch_size = 256
window_size = 50
starter_learning_rate = 0.001
learning_rate_decay = 1.0

today = datetime.today() + timedelta(0)
today_format = today.strftime('%Y%m%d')
ckpt_dir = './dmr_' + today_format


def train():
	train_data = DataIterator('alimama.txt', batch_size, 20)
	global_step = tf.Variable(0, name="global_step", trainable=False)
	learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 2000000, learning_rate_decay, staircase=True)
	# construct the model structure
	model = Model_DMR(learning_rate, global_step)
	iter = 0
	test_iter = 100
	loss_sum = 0.0
	accuracy_sum = 0.
	aux_loss_sum = 0.
	stored_arr = []

	init = tf.global_variables_initializer()
	local_init = tf.local_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)
		sess.run(local_init)
		saver = tf.train.Saver()
		for features, targets in train_data:
			loss, acc, aux_loss, prob = model.train(sess, features, targets)
			loss_sum += loss
			accuracy_sum += acc
			aux_loss_sum += aux_loss
			prob_1 = prob[:, 0].tolist()
			target_1 = targets.tolist()
			for p, t in zip(prob_1, target_1):
				stored_arr.append([p, t])
			iter += 1
			if (iter % test_iter) == 0:
				print(datetime.now().ctime())
				print('globel step=', global_step)
				print('iter: %d ----> train_loss: %.4f ---- train_accuracy: %.4f ---- train_aux_loss: %.4f ---- train_auc: %.4f' % \
					  (iter, loss_sum / test_iter, accuracy_sum / test_iter, aux_loss_sum / test_iter, calc_auc(stored_arr)))
				loss_sum = 0.0
				accuracy_sum = 0.0
				aux_loss_sum = 0.0
				stored_arr = []
				saver.save(sess, save_path=ckpt_dir)
		print("session finished.")


def eval():
	test_data = DataIterator('alimama_test.txt', batch_size, 20)
	global_step = tf.Variable(0, name="global_step", trainable=False)
	learning_rate = 0.001
	model = Model_DMR(learning_rate, global_step)
	iter = 0
	test_iter = 100
	loss_sum = 0.0
	accuracy_sum = 0.
	aux_loss_sum = 0.
	stored_arr = []

	init = tf.global_variables_initializer()
	local_init = tf.local_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)
		sess.run(local_init)
		saver = tf.train.Saver()
		saver.restore(sess, save_path=ckpt_dir)
		for features, targets in test_data:
			loss, acc, aux_loss, prob = model.calculate(sess, features, targets)
			loss_sum += loss
			accuracy_sum += acc
			aux_loss_sum += aux_loss
			prob_1 = prob[:, 0].tolist()
			target_1 = targets.tolist()
			for p, t in zip(prob_1, target_1):
				stored_arr.append([p, t])
			iter += 1
			if (iter % test_iter) == 0:
				print(datetime.now().ctime())
				print('globel step=', global_step)
				print(
				'iter: %d ----> test_loss: %.4f ---- test_accuracy: %.4f ---- test_aux_loss: %.4f ---- test_auc: %.4f' % \
				(iter, loss_sum / iter, accuracy_sum / iter, aux_loss_sum / iter, calc_auc(stored_arr)))
		print("session finished.")


if __name__ == "__main__":
	SEED = 3
	tf.set_random_seed(SEED)
	np.random.seed(SEED)
	random.seed(SEED)
	if sys.argv[1] == 'train':
		train()
	elif sys.argv[1] == 'test':
		eval()

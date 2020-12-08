import numpy as np
import json
from sklearn.manifold import TSNE
import pickle
import tensorflow as tf
import os, os.path
import pickle
import matplotlib.pyplot as plt
from functools import partial
import PIL.Image
from IPython.display import clear_output, Image, display, HTML

batch_size = 1000
max_sequence_length = 25
rnn_cell_size = 256
vocabulary_size = 8745
word_embedding_size = 300
train_size = 400000
test_size = 50000
threshold = 0.844
goodUntil = 2
display_step = train_size/batch_size#1000
epoch = 5
training_itr = display_step * epoch#1000*10
rate = 0.01
decay = 0.3
schedule = train_size/batch_size



print('TensorFlow version: {0}'.format(tf.__version__))

def importTrainingData():
    npzfile = np.load("train_and_val.npz")
    train_x = npzfile["train_x"]
    train_y = npzfile["train_y"]
    #train_y = np.reshape(train_y, [-1, 1]);
    train_y = np.resize(train_y, (train_size,1))
    train_mask = npzfile["train_mask"]
    #validation filenames follow the same pattern
    return train_x, train_y, train_mask

def importTestingData():
    npzfile = np.load("train_and_val.npz")
    #validation filenames follow the same pattern
    val_x = npzfile["val_x"]
    val_y = npzfile["val_y"]
    val_y = np.resize(val_y, (test_size,1))
    val_mask = npzfile["val_mask"]
    return val_x, val_y, val_mask

def trainRNN():
	global rnn_cell_size, batch_size, max_sequence_length, vocabulary_size, word_embedding_size, rate, decay, schedule, training_itr, display_step, threshold, goodUntil
	sequence_placeholder = tf.placeholder(tf.int64, [None, max_sequence_length])
	w_embed  = tf.Variable(tf.truncated_normal([vocabulary_size, word_embedding_size], stddev=0.001))
	rnn_input = tf.cast(tf.nn.embedding_lookup(w_embed, sequence_placeholder), tf.float64)
	#RNN
	print("rnn size")
	print(rnn_input.get_shape())
	cell = tf.contrib.rnn.LSTMCell(rnn_cell_size)
	cell = tf.contrib.rnn.DropoutWrapper(cell = cell, output_keep_prob=0.5)
	#cell = tf.nn.rnn_cell.MultiRNNCell(cells=[cell] * 4, state_is_tuple = True)
	output, last_states = tf.nn.dynamic_rnn(cell=cell, dtype=tf.float64,inputs = rnn_input)
	#mask
	mask_placeholder = tf.placeholder(tf.float64, [None, max_sequence_length])
	length = tf.cast(tf.reduce_sum(mask_placeholder, reduction_indices=1), tf.int32)
	batch_size_tf = tf.shape(output)[0]
	max_length = tf.shape(output)[1]
	out_size = int(output.get_shape()[2])
	print("output")
	print(output.get_shape())
	flat = tf.reshape(output, [-1, out_size])
	index = tf.range(0, batch_size_tf) * max_length + (length - 1)
	relevant = tf.gather(flat, index)
	print("relevant size:")
	print(relevant.get_shape())
	#output logit
	W = tf.Variable(tf.truncated_normal([rnn_cell_size, 1], dtype=tf.float64, stddev=0.001))
	W0 = tf.Variable(tf.truncated_normal([1], stddev=0.001, dtype = tf.float64))

	pred = tf.add(tf.matmul(relevant, W), W0)
	#saving: preparing model
	predict_op = tf.div(tf.nn.relu(pred), pred)
	tf.get_collection("validation_nodes")
	tf.add_to_collection("validation_nodes", sequence_placeholder)
	tf.add_to_collection("validation_nodes", mask_placeholder)
	tf.add_to_collection("validation_nodes", predict_op)

	y = tf.placeholder(tf.float64, [None,1])
	eta = tf.placeholder(tf.float64)

	#cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=pred)
	#cross_entropy *= mask_placeholder
	#cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
	#cross_entropy /= tf.reduce_sum(mask_placeholder, reduction_indices=1)
	#cost = tf.reduce_mean(cross_entropy)
	cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=pred))

	optimizer = tf.train.AdamOptimizer(learning_rate=eta).minimize(cost)
	correct_pred = tf.equal(tf.div(tf.nn.relu(pred), pred), y)
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float64))
	init = tf.global_variables_initializer()

	train_x, train_y, train_mask = importTrainingData()
	test_x, test_y, test_mask = importTestingData()

	train_acc_list = []
	test_acc_list = []
	loss_list = []
	saver = tf.train.Saver()

	#min_after_dequeue = train_size
	#capacity = min_after_dequeue + 3 * batch_size
	#batch_x, batch_y, batch_mask = tf.train.shuffle_batch([train_x, train_y, train_mask], batch_size=batch_size, capacity=capacity, enqueue_many = True, min_after_dequeue=min_after_dequeue)

	with tf.Session() as sess:
		sess.run(init)
		#coord = tf.train.Coordinator()
		#threads = tf.train.start_queue_runners(coord=coord)
		#tf.train.start_queue_runners(sess=sess)

		step = 1
		good = 0
		while step < training_itr:
			#epoch_x, epoch_y, epoch_mask = sess.run([batch_x, batch_y, batch_mask])
			choices = np.random.choice(train_size, batch_size, replace = False)
			batch_x = train_x[choices]
			batch_y = train_y[choices]
			batch_mask = train_mask[choices]
			if step % 100 == 0:
				print("Iteration:", step)
			#print batch_y[0]
			#sess.run(optimizer, feed_dict={sequence_placeholder: epoch_x, y: epoch_y, mask_placeholder: epoch_mask, eta: rate})
			sess.run(optimizer, feed_dict={sequence_placeholder: batch_x, y: batch_y, mask_placeholder: batch_mask, eta: rate})
			#onePredict = sess.run(pred, feed_dict = {sequence_placeholder: batch_x[0].reshape(1, max_sequence_length), mask_placeholder: batch_mask[0].reshape(1,max_sequence_length)})
			#print onePredict
			if step % display_step == 0:
				loss, train_acc = sess.run([cost, accuracy], feed_dict={sequence_placeholder: batch_x, y: batch_y, mask_placeholder: batch_mask})
				test_acc = sess.run(accuracy, feed_dict={sequence_placeholder: test_x, y: test_y, mask_placeholder: test_mask})
				print("Epoch: "+str(step*batch_size/train_size)+"/"+str(epoch)+ ", Loss= "+"{:.6f}".format(loss) + ", train accurary= "+"{:.5f}".format(train_acc) + ", test accuracy= "+"{:.5f}".format(test_acc))
				train_acc_list.append(train_acc)
				test_acc_list.append(test_acc)
				loss_list.append(loss)
				if test_acc > threshold:
					good = good + 1
					if good >= goodUntil:
						break
			if step % schedule == 0:
				rate = rate*decay
				print('learning rate decreased to: '+ str(rate))

			step += 1
		print("Done!")
		print("Test accuracy:")
		acc_ave = sess.run(accuracy, feed_dict={sequence_placeholder: test_x,y: test_y, mask_placeholder: test_mask})
		print("Average: "+"{:.6f}".format(acc_ave))

	    #save model
		save_path = saver.save(sess, "my_model")
	    #save weights, for plotting filters
		filters = [W.eval(), W0.eval()]
		with open('filters','wb') as fp:
			pickle.dump(filters, fp)
		embedding = sess.run(w_embed)
		with open('embedding', 'wb') as fp:
			pickle.dump(embedding, fp)
		sess.close()
	    #save accuracy lists and loss list, for plotting
		with open('train_acc','wb') as fp:
		    pickle.dump(train_acc_list, fp)
		with open('test_acc','wb') as fp:
		    pickle.dump(test_acc_list, fp)
		with open('cost','wb') as fp:
		    pickle.dump(loss_list, fp)
		#coord.request_stop()
		#coord.join(threads)

def plotAcc():
    print("Plotting accuracy figures...")
    with open ('train_acc', 'rb') as fp:
        train_acc_list = pickle.load(fp)
    with open ('test_acc', 'rb') as fp:
        test_acc_list = pickle.load(fp)
    with open ('cost', 'rb') as fp:
        loss_list = pickle.load(fp)
    train_acc = train_acc_list#[::10]
    test_acc = test_acc_list#[::10]
    loss = loss_list#[::10]

    t = list(range(len(loss)))
    #fig, ax1 = plt.subplots()
    #ax1.plot(t, loss,'b')
    #ax1.set_xlabel('Iterations')
    #ax1.set_ylabel('Cost', color='b')
    #ax1.tick_params('y', colors='b')
    #ax1.set_ylim([0.25,0.95])

    #ax2 = ax1.twinx()
    plt.plot(t, train_acc, 'r')
    plt.plot(t, test_acc,'b')
    plt.plot(t, loss, 'g')
    axes = plt.gca()
    axes.set_ylim([0.2, 0.95])
    #ax2.plot(t, train_acc,'r')
    #ax2.plot(t, test_acc,'g')
    #ax2.set_ylabel('Accuracy', color='k')
    #ax2.tick_params('y', colors='k')
    #fig.tight_layout()
    plt.legend(['Train Accurary', 'Test Accuracy', 'Loss'], loc = 5)
    plt.savefig('accuracy.pdf')
    # plt.show()

def calcEmbed():
	with open("vocab.json", "r") as f:
		vocab = json.load(f)
	s = ["monday", "tuesday", "wednesday", "thursday", "friday",
	    "saturday", "sunday", "orange", "apple", "banana", "mango",
	    "pineapple", "cherry", "fruit"]
	words = [(i, vocab[i]) for i in s]

	with open("embedding") as f:
		word_embedding_matrix = pickle.load(f)

	model = TSNE(n_components = 2, random_state = 0)
	tsne_embedding = model.fit_transform(word_embedding_matrix)
	words_vectors = tsne_embedding[np.array([item[1][0] for item in words])]
	with open("words_vectors", "wb") as f:
		pickle.dump(words_vectors, f)

def plotEmbed():
	s = ["monday", "tuesday", "wednesday", "thursday", "friday",
	    "saturday", "sunday", "orange", "apple", "banana", "mango",
	    "pineapple", "cherry", "fruit"]
	with open("words_vectors") as f:
		words_vectors = pickle.load(f)
	print(words_vectors)

	x = words_vectors[:,0]
	y = words_vectors[:,1]

	print(x[0],y[0])

	fig, ax = plt.subplots()
	ax.scatter(x, y)
	for i, txt in enumerate(s):
		ax.annotate(txt, (x[i], y[i]))
	plt.savefig('wvv.pdf')
	plt.show()
    
def main():
	trainRNN()
	plotAcc()
	# calcEmbed()
	# plotEmbed()

if __name__ == '__main__':
    main()

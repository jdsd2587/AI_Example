import tensorflow.compat.v1 as tf
import numpy as np

data = [[2, 0], [4, 0], [6, 0], [8, 1], [10, 1], [12, 1], [14, 1]]
x_data = [x_row[0] for x_row in data]
y_data = [y_row[1] for y_row in data]

a = tf.Variable(tf.random_normal([1], dtype = tf.float64, seed = 0))
b = tf.Variable(tf.random_normal([1], dtype = tf.float64, seed = 0))

y = 1/(1 + np.e**(a * x_data + b))

loss = -tf.reduce_mean(np.array(y_data) * tf.log(y) + (1 - np.array(y_data)) * tf.log(1 - y))

learning_rate = 0.5

gradient_descent = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  for i in range(60001):
    sess.run(gradient_descent)
    if i % 6000 == 0:
      print("Epoch: %.f, loss = %.4f, 기울기 a = %.4f, y 절편 = %.4f" % (i, sess.run(loss), sess.run(a), sess.run(b)))
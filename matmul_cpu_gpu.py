import tensorflow as tf
import numpy as np
import time

start = time.time()

multiplier=50

m1=500*multiplier
n1=200*multiplier
mat1 = np.arange(m1*n1,dtype='float32').reshape(m1, n1)
m2=n1
n2=400*multiplier
mat2 = np.arange(m2*n2,dtype='float32').reshape(m2, n2)

end = time.time()

print(format((end - start), '.2f')," seconds spent in numpy data prep")
print(" Matrix dimensions m1 :: m2 ",mat1.shape," :: ",mat2.shape)
start = time.time()

with tf.device('/cpu:0'):
#with tf.device('/gpu:0'):
  matrix1 = tf.Variable(mat1, name="M")
  matrix2 = tf.Variable(mat2, name="N")
  product = tf.matmul(matrix1, matrix2)
end = time.time()

print(format((end - start), '.2f')," seconds spent in TensorFlow graph prep")


start = time.time()

#with tf.Session() tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  result = sess.run([product])
  print("Done")

end = time.time()
print(format((end - start), '.2f')," seconds spent in TensorFlow Calculation")

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 23:33:34 2018

@author: kosei
"""

import time
import csv
from pandas import read_csv
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)

def weight_variable(shape,name):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape,name):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial, name=name)

class VAE:
    
    def __init__(self,
                 dim_in=28*28,
                 dim_L1=500,
                 dim_L2=500,
                 dim_L3=20):
        
        # クラス内変数の定義
        self.dim_in = dim_in
        self.dim_L1 = dim_L1
        self.dim_L2 = dim_L2
        self.dim_L3 = dim_L3
        
        # 各層のparemeterの初期化
        ## Encoder
        self.e_W1 = weight_variable([dim_in, dim_L1],'e_W1')
        self.e_b1 = bias_variable([dim_L1],'e_b1')
        self.e_W2 = weight_variable([dim_L1, dim_L2],'e_W2')
        self.e_b2 = bias_variable([dim_L2],'e_b2')
        self.e_W3_mu = weight_variable([dim_L2, dim_L3],'e_W3_mu')
        self.e_b3_mu = bias_variable([dim_L3],'e_b3_mu')
        self.e_W3_var = weight_variable([dim_L2, dim_L3],'e_W3_var')
        self.e_b3_var = bias_variable([dim_L3],'e_b3_var')
        ## Decoder
        self.d_W1 = weight_variable([dim_L3, dim_L2],'d_W1')
        self.d_b1 = bias_variable([dim_L2],'d_b1')
        self.d_W2 = weight_variable([dim_L2, dim_L1],'d_W2')
        self.d_b2 = bias_variable([dim_L1],'d_b2')
        self.d_W3 = weight_variable([dim_L1, dim_in],'d_W3')
        self.d_b3 = bias_variable([dim_in],'d_b3')
    
    def build_model(self, batch_size=1):
        # 入力
        self.x = tf.placeholder(tf.float32, [batch_size, self.dim_in])
        
        # encode
        mu, ln_var = self.encode(self.x)
        sqrt_var = tf.sqrt(tf.exp(ln_var))
        
        # 潜在変数zのサンプリング
        z = self.sample_z(mu, sqrt_var, batch_size)
        self.z = z
        
        # decode
        out_x = self.decode(z)
        
        # lossの定義
        ## KL divergence
        tcdm = tf.contrib.distributions.MultivariateNormalDiag # 分散が対角行列の多変量ガウス分布
        q_gauss = tcdm(loc=mu, scale_diag=sqrt_var)
        p_gauss = tcdm(loc=tf.zeros(shape=self.dim_L3), scale_diag=tf.ones(shape=self.dim_L3))
        kl_div = self.KL_divergence(q_gauss, p_gauss)
        ## reconstruction error
        rec_err = self.reconstruction_error(self.x, out_x)
        
        loss = tf.reduce_mean(kl_div + rec_err)
        
        return self.x,loss
        
    def KL_divergence(self, model1, model2):
        """
        sq_mean = tf.square(mu)
        var = tf.exp(ln_var)
        
        tmp = 1 + ln_var - sq_mean + var
        kl_div = -0.5*tf.reduce_sum(tmp, reduction_indices=[1])
        
        return tf.reduce_mean(kl_div)
        """
        return tf.reduce_mean(model1.kl_divergence(other=model2))
    
    def reconstruction_error(self, inp_x, out_x):
        
        # 入力と出力の2乗誤差
        sq_err = tf.reduce_sum(tf.squared_difference(inp_x, out_x), reduction_indices=[1])
        sq_err = tf.multiply(0.5, sq_err)
        
        return tf.reduce_mean(sq_err)
        
        """
        cross_entropy = -tf.reduce_sum(inp_x*tf.log(1e-10+out_x) + (1-inp_x)*tf.log(1e-10+1-out_x), reduction_indices=[1])
        return cross_entropy
        """
    
    def encode(self, inp_x):
        # 1層目
        h_enc1 = tf.nn.softplus(tf.matmul(inp_x, self.e_W1) + self.e_b1)
        # 2層目
        h_enc2 = tf.nn.softplus(tf.matmul(h_enc1, self.e_W2) + self.e_b2)
        # 3層目
        ## mean
        self.mu = tf.matmul(h_enc2, self.e_W3_mu) + self.e_b3_mu
        ## ln_var
        self.ln_var = tf.matmul(h_enc2, self.e_W3_var) + self.e_b3_var
        
        return self.mu, self.ln_var
    
    def decode(self, z):
        # 1層目
        h_dec1 = tf.nn.softplus(tf.matmul(z, self.d_W1) + self.d_b1)
        # 2層目
        h_dec2 = tf.nn.softplus(tf.matmul(h_dec1, self.d_W2) + self.d_b2)
        # 3層目
        out_x = tf.nn.sigmoid(tf.matmul(h_dec2, self.d_W3) + self.d_b3)
        
        return out_x
    
    def train(self, X, batch_size=1, epoch=1):
        # モデル構築
        inp_x, loss = self.build_model(batch_size=batch_size)
        train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)
        # セッション開始
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
        n,d = X.shape
        index = np.arange(n)
        itr = int(np.ceil(n/batch_size))
        print('学習開始')
        start = time.time()
        for i in range(epoch):
            np.random.shuffle(index)
            for j in range(itr):
                start = j*batch_size
                end = j*batch_size + batch_size
                batch_id = index[start:end]
                batch = X[batch_id, :]
                self.sess.run(train_step, feed_dict={inp_x:batch})
            
            t_loss = self.sess.run(loss, feed_dict={inp_x:batch})
            print('epoch'+str(i+1)+' 経過時間: '+str(time.time() - start))
            print('train loss: ' + str(t_loss))
        print('学習終了 時間:{}'.format(time.time() - start))
    
    def sample_z(self, mu, sqrt_var, batch_size):
        epsilon = tf.random_normal([batch_size, self.dim_L3], 0, 1, dtype=tf.float32)
        z = mu + tf.multiply(epsilon, sqrt_var)
        return z
    
    def generate_sample(self, rows=4, cols=8):
        z = self.sample_z(0, 1, rows*cols)
        sample = self.decode(z)
        sample = self.sess.run(sample)
        for index, data in enumerate(sample):
            # 画像データはrows * colsの行列上に配置
            plt.subplot(rows, cols, index + 1)
            # 軸表示は無効
            plt.axis("off")
            # データをグレースケール画像として表示
            plt.imshow(data.reshape(28,28), cmap="gray", interpolation="nearest")
        plt.savefig("generated_sample.png")
        plt.show()
    
    def to_latent_space(self, X):
        _, _ = self.build_model(batch_size=X.shape[0])
        return self.sess.run(self.mu, feed_dict={self.x:X})
    
    def write_dec_par(self):
        sess = self.sess
        
        W_vars = [x for x in tf.trainable_variables() if "d_W" in x.name]
        for var in W_vars:
            v = sess.run(var)
            f = open('decode_parameter/'+str(var.name)+'_dim'+str(self.dim_L3)+'.csv', newline='')
            writer = csv.writer(f)
            writer.writerows(v)
            f.close()
        
        b_vars = [x for x in tf.trainable_variables() if "d_b" in x.name]
        for var in b_vars:
            v = sess.run(var)
            f = open('decode_parameter/'+str(var.name)+'_dim'+str(self.dim_L3)+'.csv')
            writer = csv.writer(f)
            writer.writerow(v)
            f.close()
        
        
    def write_enc_par(self,sess):
        sess = self.sess
        
        W_vars = [x for x in tf.trainable_variables() if "e_W" in x.name]
        for var in W_vars:
            v = sess.run(var)
            f = open('encode_parameter/'+str(var.name)+'_dim'+str(self.dim_L3)+'.csv', newline='')
            writer = csv.writer(f)
            writer.writerows(v)
            f.close()
        
        b_vars = [x for x in tf.trainable_variables() if "e_b" in x.name]
        for var in b_vars:
            v = sess.run(var)
            f = open('encode_parameter/'+str(var.name)+'_dim'+str(self.dim_L3)+'.csv')
            writer = csv.writer(f)
            writer.writerow(v)
            f.close()

if(__name__ == '__main__'):
    dim_z = 2
    batch_size = 100
    epoch = 30
    sess = tf.InteractiveSession()
    inp_x = mnist.train.images
    labels = mnist.train.labels
    
    vae = VAE(dim_L3=dim_z)
    vae.train(X=inp_x, batch_size=batch_size, epoch=epoch)
    
    label = np.array([[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]], dtype=np.int)
    label = np.dot(labels, label)[:,0]
    
    index = np.array([],dtype=np.int)
    for i in range(10):
        idx = np.where(label == i)[0]
        idx = np.random.choice(idx, 100, replace=False)
        index = np.append(index, idx)
    
    mu = vae.to_latent_space(inp_x[index])
    label = label[index]
    sc = plt.scatter(mu[:,0], mu[:,1], c=label, vmax=9, vmin=0, cmap='tab10')
    for i in range(10):
        idx = np.where(label == i)[0]
        ave = np.mean(mu[idx],axis=0)
        plt.text(ave[0],ave[1],str(i))
    plt.colorbar(sc)
    plt.savefig('latent_space.png')
    plt.show()
    
    vae.generate_sample()
    

"""
imgs = mnist.test.images
labels = mnist.test.labels
digits = np.array([], dtype=np.int32)
for label in labels:
    digit = np.where(label > 0.5)
    digit = digit[0][0]
    digits = np.append(digits, digit)

digit_list = {}
for n in range(10):
    i = np.where(digits == n)
    digit_list[str(n)] = i[0]
    
def encode(x):
    W1 = read_csv('encode_parameter/W1.csv',header=None)
    b1 = read_csv('encode_parameter/b1.csv',header=None)
    W2 = read_csv('encode_parameter/W2.csv',header=None)
    b2 = read_csv('encode_parameter/b2.csv',header=None)
    W3 = read_csv('encode_parameter/W3.csv',header=None)
    b3 = read_csv('encode_parameter/b3.csv',header=None)
    
    W1 = tf.constant(W1.values, dtype=tf.float32)
    b1 = tf.constant(b1.values, dtype=tf.float32)
    W2 = tf.constant(W2.values, dtype=tf.float32)
    b2 = tf.constant(b2.values, dtype=tf.float32)
    W3 = tf.constant(W3.values, dtype=tf.float32)
    b3 = tf.constant(b3.values, dtype=tf.float32)
    
    h1 = tf.nn.softplus(tf.matmul(x,W1) + b1)
    h2 = tf.nn.softplus(tf.matmul(h1,W2) + b2)
    z = tf.matmul(h2,W3) + b3
    return z.eval()

def decode(z):
    W1 = read_csv('decode_parameter/W1.csv',header=None)
    b1 = read_csv('decode_parameter/b1.csv',header=None)
    W2 = read_csv('decode_parameter/W2.csv',header=None)
    b2 = read_csv('decode_parameter/b2.csv',header=None)
    W3 = read_csv('decode_parameter/W3.csv',header=None)
    b3 = read_csv('decode_parameter/b3.csv',header=None)
    
    W1 = tf.constant(W1.values, dtype=tf.float32)
    b1 = tf.constant(b1.values, dtype=tf.float32)
    W2 = tf.constant(W2.values, dtype=tf.float32)
    b2 = tf.constant(b2.values, dtype=tf.float32)
    W3 = tf.constant(W3.values, dtype=tf.float32)
    b3 = tf.constant(b3.values, dtype=tf.float32)
    
    z = z.astype(np.float32)
    
    h1 = tf.nn.softplus(tf.matmul(z,W1) + b1)
    h2 = tf.nn.softplus(tf.matmul(h1,W2) + b2)
    x = tf.nn.sigmoid(tf.matmul(h2,W3) + b3)
    
    return x.eval()

g_z = np.array([])
for n in range(10):
    z = encode(imgs[digit_list[str(n)]])
    g = np.mean(z, axis=0)
    g_z = np.append(g_z, g)

g_z = np.reshape(g_z, [10,20])
    
ts = np.linspace(0, 1, num=100)

for itr,t in enumerate(ts):
    for i,z_i in enumerate(g_z):
        for j,z_j in enumerate(g_z):
            z = (1-t)*z_i + t*z_j
            z = np.reshape(z, [1,20])
            data = decode(z)
            # 画像データはrows * colsの行列上に配置
            plt.subplot(10, 10, 10*i+j + 1)
            # 軸表示は無効
            plt.axis("off")
            # データをグレースケール画像として表示
            plt.imshow(data.reshape(28,28), cmap="gray", interpolation="nearest")
            print('{} : {},{}'.format(t,i+1, j+1))
    plt.savefig('generated_sample/gif_sample/'+str('%03d'%(itr+1))+'.jpg')
    plt.show()
"""
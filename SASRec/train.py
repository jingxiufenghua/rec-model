"""
Updated on Dec 20, 2020

train SASRec model

@author: Ziyao Geng(zggzy1996@163.com)
"""
import os
import tensorflow as tf
from time import time
from tensorflow.keras.optimizers import Adam

from model import SASRec
from evaluate import *
from utils import *


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__ == '__main__':
    # =============================== GPU ==============================
    # gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
    # print(gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = '5, 6'
    # ========================= Hyper Parameters =======================
    file = '../dataset/ml-1m/ratings.dat'
    trans_score = 1
    maxlen = 200
    test_neg_num = 100

    embed_dim = 50
    blocks = 2
    num_heads = 1
    ffn_hidden_unit = 64
    dropout = 0.2
    norm_training = True
    causality = False
    embed_reg = 0  # 1e-6
    K = 10

    learning_rate = 0.001
    epochs = 50
    batch_size = 512
    # ========================== Create dataset =======================
    item_fea_col, train, val, test = create_ml_1m_dataset(file, trans_score, embed_dim, maxlen, test_neg_num)

    # ============================Build Model==========================
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = SASRec(item_fea_col, blocks, num_heads, ffn_hidden_unit, dropout,
                       maxlen, norm_training, causality, embed_reg)
        model.summary()
        # =========================Compile============================
        model.compile(optimizer=Adam(learning_rate=learning_rate))

    results = []
    for epoch in range(1, epochs + 1):
        # ===========================Fit==============================
        t1 = time()
        model.fit(
            train,
            validation_data=(val, None),
            epochs=1,
            batch_size=batch_size,
        )
        # ===========================Test==============================
        t2 = time()
        if epoch % 5 == 0:
            hit_rate, ndcg = evaluate_model(model, test, K)
            print('Iteration %d Fit [%.1f s], Evaluate [%.1f s]: HR = %.4f, NDCG = %.4f, '
                  % (epoch, t2 - t1, time() - t2, hit_rate, ndcg))
            results.append([epoch + 1, t2 - t1, time() - t2, hit_rate, ndcg])
    # ============================Write============================
    pd.DataFrame(results, columns=['Iteration', 'fit_time', 'evaluate_time', 'hit_rate', 'ndcg']).\
        to_csv('log/SASRec_log_maxlen_{}_dim_{}_blocks_{}_heads_{}_K_{}_.csv'.
               format(maxlen, embed_dim, blocks, num_heads, K), index=False)
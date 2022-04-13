import tensorflow as tf
import pickle


print(tf.__version__)
new_model = tf.keras.models.load_model('/Users/songlei/Desktop/moji/Recommender-System-with-TF2.0/saved_model/din_model')

# Check its architecture
new_model.summary()
batch_size = 4096
with open("/Users/songlei/Desktop/moji/Recommender-System-with-TF2.0/dataset/dataset_moji.pkl", 'rb') as f:
    feature_columns = pickle.load(f)
    behavior_list = pickle.load(f)
    train = pickle.load(f)
    val = pickle.load(f)
    test = pickle.load(f)

test_X, test_y = test

# test_X = tf.convert_to_tensor(test_X,dtype=tf.float32)
# test_y = tf.convert_to_tensor(test_y,dtype=tf.float32)

loss, acc = new_model.evaluate(test_X, test_y, batch_size=batch_size)

# print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

# print(new_model.predict(test_images).shape)
import json
import AttrbuteSimProc
import tensorflow as tf
import numpy as np
import math

train_twi_data_js_path = '../training_data/twitter_info_list_train.js'
train_flk_data_js_path = '../training_data/flickr_info_list_train.js'

# AttrbuteSimProc.username_similarity(train_twi_data_js_path, train_flk_data_js_path,
#                                     '../training_data/username_similarity_matrix.txt')
#
# AttrbuteSimProc.location_similaritydef(train_twi_data_js_path, train_flk_data_js_path,
#                                        '../training_data/location_similarity_matrix.txt')
#
# AttrbuteSimProc.bio_similarity(train_twi_data_js_path, train_flk_data_js_path,
#                                '../training_data/bio_similarity_matrix.txt')
#
# AttrbuteSimProc.personal_website_similarity(train_twi_data_js_path, train_flk_data_js_path,
#                                 '../training_data/personal_website_similarity_matrix.txt')

# AttrbuteSimProc.profile_img_similarity_no_face(train_twi_data_js_path, train_flk_data_js_path,
#                                                '../training_data/twitter_profile_images/',
#                                                '../training_data/flickr_profile_images/',
#                                                '../training_data/profile_pic_similarity_matrix_noface.txt',
#                                                '../training_data/both_have_faces_paris.txt')

# AttrbuteSimProc.profile_img_similarity_face('../training_data/', '../training_data/both_have_faces_paris.txt',
#                                             '../training_data/profile_pic_similarity_matrix_face.txt',
#                                             160, 10, 10)
#
# face_matrix = np.loadtxt('../training_data/profile_pic_similarity_matrix_face.txt')
# noface_matrix = np.loadtxt('../training_data/profile_pic_similarity_matrix_noface.txt')
#
# sumup_profile_pic_matrix = face_matrix + noface_matrix
# np.savetxt('../training_data/profile_pic_similarity_matrix.txt', sumup_profile_pic_matrix)

username_weight = tf.Variable( tf.random_uniform([1], -1.0, 1.0), name='username_weight')


location_weight = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name='location_weight')

bio_weight = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name='bio_weight')

website_weight = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name='website_weight')

pic_weight = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name='pic_weight')

username_diag_placeholder = tf.placeholder(tf.float32, shape=(10, 1))
location_diag_placeholder = tf.placeholder(tf.float32, shape=(10, 1))
bio_diag_placeholder = tf.placeholder(tf.float32, shape=(10, 1))
website_diag_placeholder = tf.placeholder(tf.float32, shape=(10, 1))
pic_diag_placeholder = tf.placeholder(tf.float32, shape=(10, 1))

username_other_placeholder = tf.placeholder(tf.float32, shape=(10, 1))
location_other_placeholder = tf.placeholder(tf.float32, shape=(10, 1))
bio_other_placeholder = tf.placeholder(tf.float32, shape=(10, 1))
website_other_placeholder = tf.placeholder(tf.float32, shape=(10, 1))
pic_other_placeholder = tf.placeholder(tf.float32, shape=(10, 1))

ones_placeholder = tf.ones(shape=(10, 1))

representation = ones_placeholder - (tf.mul(username_weight, username_diag_placeholder - tf.mul(username_other_placeholder, 1/9.0)) +
                                     tf.mul(location_weight, location_diag_placeholder - tf.mul(location_other_placeholder, 1/9.0)) +
                                     tf.mul(website_weight, website_diag_placeholder - tf.mul(website_other_placeholder, 1/9.0)) +
                                     tf.mul(bio_weight, bio_diag_placeholder - tf.mul(bio_other_placeholder, 1/9.0)) +
                                     tf.mul(pic_weight, pic_diag_placeholder - tf.mul(pic_other_placeholder, 1/9.0))
                                     )
# total_sim_matrix_placeholder = tf.mul(username_weight, username_matrix_placeholder) + \
#     tf.mul(location_weight, location_matrix_placeholder) + tf.mul(bio_weight, bio_matrix_placeholder) + \
#     tf.mul(website_weight, website_matrix_placeholder) + tf.mul(pic_weight, pic_matrix_placeholder)

loss = tf.reduce_mean(tf.square(representation))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session()
sess.run(init)

# username similarity matrix
username_sim_matrix_11 = np.loadtxt('../training_data/username_similarity_matrix.txt')
username_sim_matrix_10 = np.zeros((10, 10))
username_diag = np.zeros((10, 1))
username_other = np.zeros((10, 1))
tmp_other = 0
for i in range(10):
    for j in range(10):
        username_sim_matrix_10[i][j] = username_sim_matrix_11[i][j]
        if i == j:
            username_diag[i][0] = username_sim_matrix_10[i][j]
        else:
            tmp_other += username_sim_matrix_10[i][j]

    username_other[i][0] = tmp_other
    tmp_other = 0


# location similarity matrix
location_diag = np.zeros((10, 1))
location_sim_matrix_11 = np.loadtxt('../training_data/location_similarity_matrix.txt')
location_sim_matrix_10 = np.zeros((10, 10))
location_other = np.zeros((10, 1))
tmp_other = 0
for i in range(10):
    for j in range(10):
        location_sim_matrix_10[i][j] = location_sim_matrix_11[i][j]
        if i == j:
            location_diag[i][0] = location_sim_matrix_10[i][j]
        else:
            tmp_other += location_sim_matrix_10[i][j]

    location_other[i][0] = tmp_other
    tmp_other = 0


# bio similarity matrix
bio_sim_matrix_11 = np.loadtxt('../training_data/bio_similarity_matrix.txt')
bio_sim_matrix_10 = np.zeros((10, 10))
bio_diag = np.zeros((10, 1))
bio_other = np.zeros((10, 1))
tmp_other = 0
for i in range(10):
    for j in range(10):
        bio_sim_matrix_10[i][j] = bio_sim_matrix_11[i][j]
        if i == j:
            bio_diag[i][0] = bio_sim_matrix_10[i][j]
        else:
            tmp_other += bio_sim_matrix_10[i][j]

    bio_other[i][0] = tmp_other
    tmp_other = 0

# website similarity matrix
web_sim_matrix_11 = np.loadtxt('../training_data/personal_website_similarity_matrix.txt')
web_sim_matrix_10 = np.zeros((10, 10))
web_diag = np.zeros((10, 1))
web_other = np.zeros((10, 1))
tmp_other = 0
for i in range(10):
    for j in range(10):
        web_sim_matrix_10[i][j] = web_sim_matrix_11[i][j]
        if i == j:
            web_diag[i][0] = web_sim_matrix_10[i][j]

        else:
            tmp_other += web_sim_matrix_10[i][j]

    web_other[i][0] = tmp_other
    tmp_other = 0

# pic similarity matrix
pic_sim_matrix_11 = np.loadtxt('../training_data/profile_pic_similarity_matrix.txt')
pic_sim_matrix_10 = np.zeros((10, 10))
pic_diag = np.zeros((10, 1))
pic_other = np.zeros((10, 1))
tmp_other = 0
for i in range(10):
    for j in range(10):
        pic_sim_matrix_10[i][j] = pic_sim_matrix_11[i][j]
        if i == j:
            pic_diag[i][0] = pic_sim_matrix_10[i][j]

        else:
            tmp_other += pic_sim_matrix_10[i][j]

    pic_other[i][0] = tmp_other
    tmp_other = 0

feed_dict = \
    {
        username_diag_placeholder: username_diag,
        location_diag_placeholder: location_diag,
        website_diag_placeholder: web_diag,
        bio_diag_placeholder: bio_diag,
        pic_diag_placeholder: pic_diag,

        username_other_placeholder: username_other,
        location_other_placeholder: location_other,
        website_other_placeholder: web_other,
        bio_other_placeholder: bio_other,
        pic_other_placeholder: pic_other,
    }

# Training.
print 'Training the data......'
weights_info = dict()
for step in range(20001):
    _, c = sess.run([train, loss], feed_dict=feed_dict)
    print 'cost is %f' % c
    if c <= 0.108886:

        print sess.run(username_weight)[0]
        print sess.run(location_weight)[0]
        print sess.run(bio_weight)[0]
        print sess.run(website_weight)[0]
        print sess.run(pic_weight)[0]
        break

print 'Finish Training'

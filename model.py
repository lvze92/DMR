# coding: utf-8

from utils import *

# user feature size
user_size = 1141730
cms_segid_size = 97
cms_group_id_size = 13
final_gender_code_size = 3
age_level_size = 7
pvalue_level_size = 4
shopping_level_size = 4
occupation_size = 3
new_user_class_level_size = 5

# item feature size
adgroup_id_size = 846812
cate_size = 12978
campaign_id_size = 423437
customer_size = 255876
brand_size = 461529

# context feature size
btag_size = 5
pid_size = 2

# embedding size
main_embedding_size = 32
other_embedding_size = 8


class Model(object):
    def __init__(self, lr, global_step):
        # input
        with tf.name_scope('Inputs'):
            self.feature_ph = tf.placeholder(tf.float32, [None, None], name='feature_ph')
            self.target_ph = tf.placeholder(tf.float32, [None, ], name='target_ph')

            self.btag_his = tf.cast(self.feature_ph[:, 0:50], tf.int32)
            self.cate_his = tf.cast(self.feature_ph[:, 50:100], tf.int32)
            self.brand_his = tf.cast(self.feature_ph[:, 100:150], tf.int32)
            self.mask = tf.cast(self.feature_ph[:, 150:200], tf.int32)
            self.match_mask = tf.cast(self.feature_ph[:, 200:250], tf.int32)

            self.uid = tf.cast(self.feature_ph[:, 250], tf.int32)
            self.cms_segid = tf.cast(self.feature_ph[:, 251], tf.int32)
            self.cms_group_id = tf.cast(self.feature_ph[:, 252], tf.int32)
            self.final_gender_code = tf.cast(self.feature_ph[:, 253], tf.int32)
            self.age_level = tf.cast(self.feature_ph[:, 254], tf.int32)
            self.pvalue_level = tf.cast(self.feature_ph[:, 255], tf.int32)
            self.shopping_level = tf.cast(self.feature_ph[:, 256], tf.int32)
            self.occupation = tf.cast(self.feature_ph[:, 257], tf.int32)
            self.new_user_class_level = tf.cast(self.feature_ph[:, 258], tf.int32)

            self.mid = tf.cast(self.feature_ph[:, 259], tf.int32)
            self.cate_id = tf.cast(self.feature_ph[:, 260], tf.int32)
            self.campaign_id = tf.cast(self.feature_ph[:, 261], tf.int32)
            self.customer = tf.cast(self.feature_ph[:, 262], tf.int32)
            self.brand = tf.cast(self.feature_ph[:, 263], tf.int32)
            self.price = tf.expand_dims(tf.cast(self.feature_ph[:, 264], tf.float32), 1)

            self.pid = tf.cast(self.feature_ph[:, 265], tf.int32)


        # Embedding layer
        with tf.name_scope('Embedding_layer'):
            self.uid_embeddings_var = tf.get_variable("uid_embedding_var", [user_size, main_embedding_size])
            tf.summary.histogram('uid_embeddings_var', self.uid_embeddings_var)
            self.uid_batch_embedded = tf.nn.embedding_lookup(self.uid_embeddings_var, self.uid)

            self.mid_embeddings_var = tf.get_variable("mid_embedding_var", [adgroup_id_size, main_embedding_size])
            tf.summary.histogram('mid_embeddings_var', self.mid_embeddings_var)
            self.mid_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid)

            self.cat_embeddings_var = tf.get_variable("cat_embedding_var", [cate_size, main_embedding_size])
            tf.summary.histogram('cat_embeddings_var', self.cat_embeddings_var)
            self.cat_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.cate_id)
            self.cat_his_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.cate_his)

            self.brand_embeddings_var = tf.get_variable("brand_embedding_var", [brand_size, main_embedding_size])
            self.brand_batch_embedded = tf.nn.embedding_lookup(self.brand_embeddings_var, self.brand)
            self.brand_his_batch_embedded = tf.nn.embedding_lookup(self.brand_embeddings_var, self.brand_his)

            self.btag_embeddings_var = tf.get_variable("btag_embedding_var", [btag_size, other_embedding_size])
            self.btag_his_batch_embedded = tf.nn.embedding_lookup(self.btag_embeddings_var, self.btag_his)
            self.dm_btag_embeddings_var = tf.get_variable("dm_btag_embedding_var", [btag_size, other_embedding_size])
            self.dm_btag_his_batch_embedded = tf.nn.embedding_lookup(self.dm_btag_embeddings_var, self.btag_his)

            self.campaign_id_embeddings_var = tf.get_variable("campaign_id_embedding_var", [campaign_id_size, main_embedding_size])
            self.campaign_id_batch_embedded = tf.nn.embedding_lookup(self.campaign_id_embeddings_var, self.campaign_id)

            self.customer_embeddings_var = tf.get_variable("customer_embedding_var", [customer_size, main_embedding_size])
            self.customer_batch_embedded = tf.nn.embedding_lookup(self.customer_embeddings_var, self.customer)

            self.cms_segid_embeddings_var = tf.get_variable("cms_segid_embedding_var", [cms_segid_size, other_embedding_size])
            self.cms_segid_batch_embedded = tf.nn.embedding_lookup(self.cms_segid_embeddings_var, self.cms_segid)

            self.cms_group_id_embeddings_var = tf.get_variable("cms_group_id_embedding_var", [cms_group_id_size, other_embedding_size])
            self.cms_group_id_batch_embedded = tf.nn.embedding_lookup(self.cms_group_id_embeddings_var, self.cms_group_id)

            self.final_gender_code_embeddings_var = tf.get_variable("final_gender_code_embedding_var", [final_gender_code_size, other_embedding_size])
            self.final_gender_code_batch_embedded = tf.nn.embedding_lookup(self.final_gender_code_embeddings_var, self.final_gender_code)

            self.age_level_embeddings_var = tf.get_variable("age_level_embedding_var", [age_level_size, other_embedding_size])
            self.age_level_batch_embedded = tf.nn.embedding_lookup(self.age_level_embeddings_var, self.age_level)

            self.pvalue_level_embeddings_var = tf.get_variable("pvalue_level_embedding_var", [pvalue_level_size, other_embedding_size])
            self.pvalue_level_batch_embedded = tf.nn.embedding_lookup(self.pvalue_level_embeddings_var, self.pvalue_level)

            self.shopping_level_embeddings_var = tf.get_variable("shopping_level_embedding_var", [shopping_level_size, other_embedding_size])
            self.shopping_level_batch_embedded = tf.nn.embedding_lookup(self.shopping_level_embeddings_var, self.shopping_level)

            self.occupation_embeddings_var = tf.get_variable("occupation_embedding_var", [occupation_size, other_embedding_size])
            self.occupation_batch_embedded = tf.nn.embedding_lookup(self.occupation_embeddings_var, self.occupation)

            self.new_user_class_level_embeddings_var = tf.get_variable("new_user_class_level_embedding_var", [new_user_class_level_size, other_embedding_size])
            self.new_user_class_level_batch_embedded = tf.nn.embedding_lookup(self.new_user_class_level_embeddings_var, self.new_user_class_level)

            self.pid_embeddings_var = tf.get_variable("pid_embedding_var", [pid_size, other_embedding_size])
            self.pid_batch_embedded = tf.nn.embedding_lookup(self.pid_embeddings_var, self.pid)

            self.user_feat = tf.concat([self.uid_batch_embedded, self.cms_segid_batch_embedded, self.cms_group_id_batch_embedded, self.final_gender_code_batch_embedded, self.age_level_batch_embedded, self.pvalue_level_batch_embedded, self.shopping_level_batch_embedded, self.occupation_batch_embedded, self.new_user_class_level_batch_embedded], -1)
            self.item_his_eb = tf.concat([self.cat_his_batch_embedded, self.brand_his_batch_embedded], -1)
            self.item_his_eb_sum = tf.reduce_sum(self.item_his_eb, 1)
            self.item_feat = tf.concat([self.mid_batch_embedded, self.cat_batch_embedded, self.brand_batch_embedded, self.campaign_id_batch_embedded, self.customer_batch_embedded, self.price], -1)
            self.item_eb = tf.concat([self.cat_batch_embedded, self.brand_batch_embedded], -1)
            self.context_feat = self.pid_batch_embedded

            self.lr = lr
            self.global_step = global_step


    def build_fcn_net(self, inp):
        inp = tf.layers.batch_normalization(inputs=inp, name='bn_inp', training=True)
        dnn0 = tf.layers.dense(inp, 512, activation=None, name='f0')
        dnn0 = prelu(dnn0, 'prelu0')
        dnn1 = tf.layers.dense(dnn0, 256, activation=None, name='f1')
        dnn1 = prelu(dnn1, 'prelu1')
        dnn2 = tf.layers.dense(dnn1, 128, activation=None, name='f2')
        dnn2 = prelu(dnn2, 'prelu2')
        dnn3 = tf.layers.dense(dnn2, 1, activation=None, name='f3')
        self.y_hat = tf.nn.sigmoid(dnn3)

        with tf.name_scope('Metrics'):
            if self.target_ph is not None:
                # Cross-entropy loss and optimizer initialization
                ctr_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.target_ph, logits=tf.reduce_sum(dnn3, 1)))
                self.ctr_loss = ctr_loss
                self.loss = ctr_loss + self.aux_loss
                tf.summary.scalar('loss', self.loss)

                # Accuracy metric
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
                tf.summary.scalar('accuracy', self.accuracy)

            self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            with tf.control_dependencies(self.update_ops):
                self.training_op = self.optimizer.minimize(self.loss, global_step=self.global_step)



    def train(self, sess, features, targets):
        loss, accuracy, aux_loss, _, probs = sess.run([self.loss, self.accuracy, self.aux_loss, self.training_op, self.y_hat], feed_dict={
            self.feature_ph: features,
            self.target_ph: targets
        })
        return loss, accuracy, aux_loss, probs

    def calculate(self, sess, features, targets):
        loss, accuracy, aux_loss, probs = sess.run([self.loss, self.accuracy, self.aux_loss, self.y_hat], feed_dict={
            self.feature_ph: features,
            self.target_ph: targets
        })
        return loss, accuracy, aux_loss, probs


class Model_DMR(Model):
    def __init__(self, lr, global_step):
        super(Model_DMR, self).__init__(lr, global_step)

        self.position_his = tf.range(50)
        self.position_embeddings_var = tf.get_variable("position_embeddings_var", [50, other_embedding_size])
        self.position_his_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his)  # T,E
        self.position_his_eb = tf.tile(self.position_his_eb, [tf.shape(self.mid)[0], 1])  # B*T,E
        self.position_his_eb = tf.reshape(self.position_his_eb, [tf.shape(self.mid)[0], -1, self.position_his_eb.get_shape().as_list()[1]])  # B,T,E

        self.dm_position_his = tf.range(50)
        self.dm_position_embeddings_var = tf.get_variable("dm_position_embeddings_var", [50, other_embedding_size])
        self.dm_position_his_eb = tf.nn.embedding_lookup(self.dm_position_embeddings_var, self.dm_position_his)  # T,E
        self.dm_position_his_eb = tf.tile(self.dm_position_his_eb, [tf.shape(self.mid)[0], 1])  # B*T,E
        self.dm_position_his_eb = tf.reshape(self.dm_position_his_eb, [tf.shape(self.mid)[0], -1, self.dm_position_his_eb.get_shape().as_list()[1]])  # B,T,E

        self.position_his_eb = tf.concat([self.position_his_eb, self.btag_his_batch_embedded], -1)
        self.dm_position_his_eb = tf.concat([self.dm_position_his_eb, self.dm_btag_his_batch_embedded], -1)

        # User-to-Item Network
        with tf.name_scope('u2i_net'):
            dm_item_vectors = tf.get_variable("dm_item_vectors", [cate_size, main_embedding_size])
            tf.summary.histogram('dm_item_vectors', dm_item_vectors)
            dm_item_biases = tf.get_variable('dm_item_biases', [cate_size], initializer=tf.zeros_initializer(), trainable=False)
            # Auxiliary Match Network
            self.aux_loss, dm_user_vector, scores = deep_match(self.item_his_eb, self.dm_position_his_eb, self.mask, tf.cast(self.match_mask, tf.float32), self.cate_his, main_embedding_size, dm_item_vectors, dm_item_biases, cate_size)
            self.aux_loss *= 0.1
            dm_item_vec = tf.nn.embedding_lookup(dm_item_vectors, self.cate_id)  # B,E
            rel_u2i = tf.reduce_sum(dm_user_vector * dm_item_vec, axis=-1, keep_dims=True)  # B,1
            self.rel_u2i = rel_u2i

        # Item-to-Item Network
        with tf.name_scope('i2i_net'):
            att_outputs, alphas, scores_unnorm = dmr_fcn_attention(self.item_eb, self.item_his_eb, self.position_his_eb, self.mask)
            tf.summary.histogram('att_outputs', alphas)
            rel_i2i = tf.expand_dims(tf.reduce_sum(scores_unnorm, [1,2]), -1)
            self.rel_i2i = rel_i2i
            self.scores = tf.reduce_sum(alphas, 1)

        inp = tf.concat([self.user_feat, self.item_feat, self.context_feat, self.item_his_eb_sum,self.item_eb * self.item_his_eb_sum, rel_u2i, rel_i2i, att_outputs], -1)
        self.build_fcn_net(inp)



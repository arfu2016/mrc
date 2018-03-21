"""
@Project   : DuReader
@Module    : match_layer_ref.py
@Author    : Deco [deco@cubee.com]
@Created   : 3/21/18 1:13 PM
@Desc      : This module implements the core layer of Match-LSTM
"""

import tensorflow as tf
import tensorflow.contrib as tc


class MatchLSTMLayer:
    """
    Implements the Match-LSTM layer, which attend to the question dynamically
    in a LSTM fashion.
    Before this layer, there are passage and question preprocessing layers
    """
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
        # hidden size of the layer

    def match(self, passage_encodes, question_encodes, p_length, q_length):
        """
        Match the passage_encodes with question_encodes
        using Match-LSTM algorithm
        Args:
            passage_encodes: 从passage preprocessing layers得到的outputs,
              在调用dynamic_rnn的时候会用到，作为输入; 比如，可以用bi-lstm
              做preprocessing, it could be a l*p matrix
            question_encodes: 从question preprocessing layers得到的outputs，
              在生成lstm cell with attention的时候会用到
            p_length: passage_encodes的长度，在调用dynamic_rnn的时候会用到，
              作为输入
        """
        with tf.variable_scope('match_lstm'):
            # 命名空间
            cell_fw = MatchLSTMAttnCell(self.hidden_size, question_encodes)
            # 带attention的向前的cell, 要带上attention，必须把 question_encodes
            # 传进去，就好像通过装饰器或者闭包产生了一个新的实例一样
            cell_bw = MatchLSTMAttnCell(self.hidden_size, question_encodes)
            # 带attention的向后的cell
            # cell_fw == cell_bw, but cell_fw is not cell_bw
            outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                             inputs=passage_encodes,
                                                             sequence_length=p_length,
                                                             dtype=tf.float32)
            # 在这个动态产生rnn网络中，cell_fw, cell_bw都是带着question attention
            # 的cell对象，最终也是得到outputs和state
            match_outputs = tf.concat(outputs, 2)
            # 在实现上把两个方向的lstm的输出连接在一起，而不是相加
            state_fw, state_bw = state
            c_fw, h_fw = state_fw
            c_bw, h_bw = state_bw
            match_state = tf.concat([h_fw, h_bw], 1)
            # 挑出hidden state，舍弃cell state
        return match_outputs, match_state
# 输出依然是outputs和state，是这个match lstm的输出


class MatchLSTMAttnCell(tc.rnn.LSTMCell):
    """
    Implements the Match-LSTM attention cell
    inherits tc.rnn.LSTMCell
    """
    def __init__(self, num_units, context_to_attend):
        """
        Args:
            num_units: size of the hidden layer
            context_to_attend: question encodes after the preprocessing of
              question lstm layer, it could be a l*q matrix

        """
        super().__init__(num_units, state_is_tuple=True)
        # 返回cell state和hidden state
        self.context_to_attend = context_to_attend
        self.fc_context = tc.layers.fully_connected(self.context_to_attend,
                                                    num_outputs=self._num_units,
                                                    activation_fn=None)
        # 把question encodes线性组合，得到新的向量或矩阵，
        # 转换用到的是一个全连接神经网络
        # Wq*Hq, Hq就是question encodes, Wq就是这个神经网络的系数
        # 注意用的是layers.fully_connected，if Hq is a l*q matrix, then after
        # the matrix multiplication, l may be changed to self._num_units,
        # q stays unchanged

    def __call__(self, inputs, state, scope=None):
        # 定义实例调用所执行的函数
        # 参数没毛病
        # 在用dynamic_rnn调用这个类的对象时，需要执行本函数
        # 具体怎样实现，需要看dynamic_rnn的代码
        (c_prev, h_prev) = state
        # 在使用dynamic_rnn时，不同的cell之间需要相互通信，就要用到state
        # 在dynamic_rnn使用每个cell时，每个cell的inputs也是需要的
        # 这里的inputs和state都是针对一个cell的输入
        with tf.variable_scope(scope or type(self).__name__):
            # 如果scope是None，用到的命名空间就是这个类的名字
            ref_vector = tf.concat([inputs, h_prev], -1)
            G = tf.tanh(self.fc_context
                        + tf.expand_dims(tc.layers.fully_connected(ref_vector,
                                                                   num_outputs=self._num_units,
                                                                   activation_fn=None), 1))
            # 可以参考match lstm的论文，就知道G是怎样计算的
            # 在论文中，inputs的系数是Wp，h_prev的系数是Wr，这里相比论文加了一个额外的
            # 假设，Wp == Wr，这样可以减少要拟合的参数的个数，Wp或者Wr就是上面这个全连接
            # 网络的系数；因为有这个假设，所以可以把inputs和h_prev连起来
            # tf.expand_dims的作用是把向量重复n次，变成一个矩阵
            # G is a l*q matrix
            logits = tc.layers.fully_connected(G, num_outputs=1,
                                               activation_fn=None)
            # 注意用的是layers.fully_connected，l变成了1，q不变，得到的是
            # 1*q vector

            scores = tf.nn.softmax(logits, 1)
            # 对线性组合进行softmax操作
            attended_context = tf.reduce_sum(self.context_to_attend * scores,
                                             axis=1)
            # 利用attention分数对questoin encodes进行加权变换
            new_inputs = tf.concat([inputs, attended_context,
                                    inputs - attended_context,
                                    inputs * attended_context],
                                   -1)
            # https://www.tensorflow.org/api_docs/python/tf/concat
            # Concatenates tensors along one dimension.
            # 把passage encodes的inputs和question encodes的attended_context
            # 进行组合后得到一个matrix
            # 论文中给出的是把inputs和attended_context再经过一次lstm，这里的操作
            # 和经过一次lstm的操作也许是等价的

            return super().__call__(new_inputs, state, scope)

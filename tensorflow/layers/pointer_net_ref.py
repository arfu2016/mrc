"""
@Project   : DuReader
@Module    : pointer_net_ref.py
@Author    : Deco [deco@cubee.com]
@Created   : 3/21/18 3:39 PM
@Desc      : This module implements the Pointer Network for selecting
answer spans, as described in:
https://openreview.net/pdf?id=B1-q5Pqxl
lstm是三维的，最基本单位是hidden unit，一般情况下，一个hidden unit对应一个cell unit，
但二者的数目也可以不同，通过神经网络做转换即可（矩阵乘法）。单个hidden unit有三个输入，
分别是cell state，hidden state和input，有两个相等的输出，分别是hidden state和output。
对于整个三维lstm，输出维度是l×q，hidden state维度是l×number_layers。l是hidden size，
理论上讲，不同的lstm layer的hidden units数目可以是不同的，但实际应用中往往是相同的，
减少要拟合的参数的个数。input的vector dimension理论上与hidden units数目也是不同的，
但在实际应用中往往也取成相同的，这样转换层也相对简单。
当然了，神经网络的设计是灵活的，对于hidden state，维度也不一定用l×number_layers，
也可以是l×1，这样的话，使用hidden state的lstm就可以只用l×1的hidden state，也就是
只有一层lstm接受hidden state，其他的lstm的hidden state用0默认填满。
"""

import tensorflow as tf
import tensorflow.contrib as tc


def custom_dynamic_rnn(cell, inputs, inputs_len, initial_state=None):
    """
    对dynamic_rnn进行改动与重载
    这是相当基本的一个函数，一般不需要改动和重载
    通过重载这个函数，可以看到lstm在x轴上是如何延伸的
    Implements a dynamic rnn that can store scores in the pointer network,
    the reason why we implements this is that the raw_rnn or dynamic_rnn
    function in Tensorflow
    seem to require the hidden unit and memory unit has the same dimension,
    and we cannot
    store the scores directly in the hidden unit.
    hidden unit: ht, hidden state
    memory unit: ct, memory state
    对于单层(单个)lstm cell，hidden unit和memory unit是多维的，二者不一定相等
    有可能最初进来的数目就是不同的，但大多数情况下，二者是相同的
    理论上，没2个hidden unit对应着1个memory unit也是可能的, 2个hidden unit的输入
    合并到1个memory unit的输出中去
    参照：https://www.quora.com/What-is-the-meaning-of-%E2%80%9CThe-number-of-units-in-the-LSTM-cell
    对于单个lstm cell，outputs和hidden units是相同的，但对于整个rnn network，
    outputs和hidden units是不同的，outputs是l*p的维度，hidden units是l的维度？
    或者是l*num_layers的维度？
    Args:
        cell: RNN cell
        inputs: the input sequence to rnn
        inputs_len: valid length
        initial_state: initial_state of the cell, including cell state and
          hidden state?
          if initial_state is None, we make the initial state 0's?
    Returns:
        outputs and state
    """
    batch_size = tf.shape(inputs)[0]
    max_time = tf.shape(inputs)[1]
    # inputs has a shape of batch_size x max_time x input_size in which
    # max_time is the number of steps in the longest sequence
    # (but all sequences could be of the same length)
    # sequence_length is a vector of size batch_size in which each element
    # gives the length of each sequence in the batch
    # leave it as default if all your sequences are of the same size.

    inputs_ta = tf.TensorArray(dtype=tf.float32, size=max_time)
    inputs_ta = inputs_ta.unstack(tf.transpose(inputs, [1, 0, 2]))
    emit_ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)
    t0 = tf.constant(0, dtype=tf.int32)
    # 用到了tf.constant，此处没有用tf.placeholder
    if initial_state is not None:
        s0 = initial_state
    else:
        s0 = cell.zero_state(batch_size, dtype=tf.float32)
        # 如果initial_state is None，其实是用0作为initial_state
    f0 = tf.zeros([batch_size], dtype=tf.bool)

    def loop_fn(t, prev_s, emit_ta, finished):
        """
        the loop function of rnn
        """
        cur_x = inputs_ta.read(t)
        # 用到了函数外的变量
        scores, cur_state = cell(cur_x, prev_s)
        # cell是传进来的cell实例，此处调用实例，实际上是调用类的__call__() method
        # 此处调用的cell是一种特殊的cell，其__call__() method会返回scores
        # 也会返回cell state和hidden state

        # copy through
        scores = tf.where(finished, tf.zeros_like(scores), scores)

        if isinstance(cell, tc.rnn.LSTMCell):
            # 如果是LSTMCell，该如何处理
            cur_c, cur_h = cur_state
            prev_c, prev_h = prev_s
            cur_state = tc.rnn.LSTMStateTuple(tf.where(finished, prev_c, cur_c),
                                              tf.where(finished, prev_h, cur_h))
            # 把previous state和current state合并成current state
        else:
            # 其他情况下该如何处理
            # 对于非lstm，没有cell state，所以不需要把hidden state取出来
            # 直接赋值即可，但要用到这一句，要定义PointerNetLSTMCell类似的东西
            cur_state = tf.where(finished, prev_s, cur_state)

        emit_ta = emit_ta.write(t, scores)
        finished = tf.greater_equal(t + 1, inputs_len)
        return [t + 1, cur_state, emit_ta, finished]
    # 把scores记录在emit_ta当中

    _, state, emit_ta, _ = tf.while_loop(
        cond=lambda _1, _2, _3, finished: tf.logical_not(tf.reduce_all(finished)),
        body=loop_fn,  # 循环的是这个函数
        loop_vars=(t0, s0, emit_ta, f0),
        parallel_iterations=32,
        swap_memory=False)

    outputs = tf.transpose(emit_ta.stack(), [1, 0, 2])
    return outputs, state
# state相比重载前略有改动，但outputs保存了新的东西，是emit_ta中的scores,
# 也就是网络对passage encodes进行attention之后的输出结果


def attend_pooling(pooling_vectors, ref_vector, hidden_size, scope=None):
    """
    Applies attend pooling to a set of vectors according to a reference vector.
    max pooling的目的是减少参数个数，vector发生了变换，维度都变掉了
    attend pooling前后，vector维度不变，但矩阵本身发生了变换
    此处attend_pooling是用来对question encodes来做矩阵变换，但使用的ref_vector是
    随机给的，或者是需要拟合的
    Args:
        pooling_vectors: the vectors to pool
        ref_vector: the reference vector
        hidden_size: the hidden size for attention function，在计算score的中间步骤中，
          要用到一个维度，把pooling_vectors的维度变成hidden_size的维度
        scope: score name
    Returns:
        the pooled vector
    """
    with tf.variable_scope(scope or 'attend_pooling'):
        U = tf.tanh(tc.layers.fully_connected(pooling_vectors,
                                              num_outputs=hidden_size,
                                              activation_fn=None,
                                              biases_initializer=None)
                    + tc.layers.fully_connected(tf.expand_dims(ref_vector, 1),
                                                num_outputs=hidden_size,
                                                activation_fn=None))
        # 先做矩阵相乘，然后取tanh
        logits = tc.layers.fully_connected(U, num_outputs=1,
                                           activation_fn=None)
        # 进行维度变换，hidden_size的维度缩减为1
        scores = tf.nn.softmax(logits, 1)
        # 得到attend的分数
        pooled_vector = tf.reduce_sum(pooling_vectors * scores, axis=1)
        # 所谓attend pooling，就是根据每一个位置的attend score, 对matrix进行加权变换
    return pooled_vector


class PointerNetLSTMCell(tc.rnn.LSTMCell):
    """
    Implements the Pointer Network Cell
    自定义的lstm cell，
    """
    def __init__(self, num_units, context_to_point):
        """num_units其实是hidden units的个数，一般来说memory units也是同样数目"""
        super().__init__(num_units, state_is_tuple=True)
        self.context_to_point = context_to_point
        self.fc_context = tc.layers.fully_connected(self.context_to_point,
                                                    num_outputs=self._num_units,
                                                    activation_fn=None)
        # 为操作attend函数做准备，表明这个lstm cell被调用时返回的是attend操作之后的结果

    def __call__(self, inputs, state, scope=None):
        (c_prev, m_prev) = state
        # 此处接受的state是previous state
        with tf.variable_scope(scope or type(self).__name__):
            U = tf.tanh(self.fc_context
                        + tf.expand_dims(tc.layers.fully_connected(m_prev,
                                                                   num_outputs=self._num_units,
                                                                   activation_fn=None),
                                         1))
            logits = tc.layers.fully_connected(U, num_outputs=1,
                                               activation_fn=None)
            scores = tf.nn.softmax(logits, 1)
            attended_context = tf.reduce_sum(self.context_to_point * scores,
                                             axis=1)
            # attend pooling之后的结果
            lstm_out, lstm_state = super().__call__(attended_context, state)
            # 把attended_context作为input调用LSTMCell的__call__
            # 返回的是output和state
        return tf.squeeze(scores, -1), lstm_state
    # 重载__call__后最终返回的是scores和state


class PointerNetDecoder:
    """
    Implements the Pointer Network
    """
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def decode(self, passage_vectors, question_vectors, init_with_question=True):
        """
        decode是一层lstm（或多层lstm），一层lstm有三个输入，cell state、hidden state
        和input，在下面的实现中，question_vectors充当cell state、hidden state，
        但维度不匹配，所以需要做额外的转换，因为只预测start和end，所以input使用
        fake input，lstm的输出对passage_vectors进行attend，拿到attend概率分布，
        就是最后需要输出的值
        Use Pointer Network to compute the probabilities of each position
        to be start and end of the answer
        Args:
            passage_vectors: the encoded passage vectors, 是第一次attend pooling后的？
            question_vectors: the encoded question vectors
            init_with_question: if set to be true,
                             we will use the question_vectors to init the state of Pointer Network
              最初的hidden state用question_vectors来初始化
        Returns:
            the probs of evary position to be start and end of the answer
            维度： 2*p?
        """
        with tf.variable_scope('pn_decoder'):
            fake_inputs = tf.zeros([tf.shape(passage_vectors)[0], 2, 1])
            # not used
            sequence_len = tf.tile([2], [tf.shape(passage_vectors)[0]])
            if init_with_question:
                random_attn_vector = tf.Variable(tf.random_normal([1,
                                                                   self.hidden_size]),
                                                 trainable=True,
                                                 name="random_attn_vector")
                # 为什么用tf.Variable而不是tf.constant或者tf.placeholder
                # 并不是需要拟合的变量啊
                pooled_question_rep = tc.layers.fully_connected(
                    attend_pooling(question_vectors, random_attn_vector,
                                   self.hidden_size),
                    num_outputs=self.hidden_size, activation_fn=None
                )
                # 在内部把question_vectors的维度转成self.hidden_size，也就是hidden
                # units的数目，其实也可以在外部转换之后再作为state输入
                init_state = tc.rnn.LSTMStateTuple(pooled_question_rep,
                                                   pooled_question_rep)
                # 对hidden state进行初始化？在初始化钱要对question_vectors进行
                # attend pooling，attend所使用的reference vector是随机得到的
                # 之前在对passage进行attend pooling时，所使用的reference vector是
                # question encodes的矩阵，维度是l*q
            else:
                init_state = None
            with tf.variable_scope('fw'):
                fw_cell = PointerNetLSTMCell(self.hidden_size, passage_vectors)
                # 此处self.hidden_size是PointerNetLSTMCell的number of hidden units
                # 这里得到的是单层的lstm cell, 如果要变成多层的cell,
                # 需要调用tc.rnn.MultiRNNCell
                fw_outputs, _ = custom_dynamic_rnn(fw_cell, fake_inputs,
                                                   sequence_len, init_state)
                # 对于向前的lstm，输出的outputs是passage的这个位置是answer开始和结束
                # 的分数
            with tf.variable_scope('bw'):
                bw_cell = PointerNetLSTMCell(self.hidden_size, passage_vectors)
                bw_outputs, _ = custom_dynamic_rnn(bw_cell, fake_inputs,
                                                   sequence_len, init_state)
                # 向后的lstm
            start_prob = (fw_outputs[0:, 0, 0:] + bw_outputs[0:, 1, 0:]) / 2
            end_prob = (fw_outputs[0:, 1, 0:] + bw_outputs[0:, 0, 0:]) / 2
            # 对前后lstm进行平均
            # 此处的两个lstm应该是不必要的，有一个lstm应该就够了，否则的话，两个lstm其实
            # 是一样的，导致start_prob和end_prob相等
            # 第一个是start, 第二个是end；第一个是end，第二个是start；第一个是一端，
            # 第二个是另一端
            # 给定start，看后边的词，或者一定范围内
            return start_prob, end_prob
            # 返回的start_prob, end_prob在拟合的时候会用到，可能要用cross entropy公式。
            # 也有可能用来评估生成句子的更复杂的量度
            # 在evaluate的时候也会用到，用来得到最后的答案。

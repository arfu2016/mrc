"""
@Project   : DuReader
@Module    : basic_rnn_ref.py
@Author    : Deco [deco@cubee.com]
@Created   : 3/21/18 10:06 AM
@Desc      : This module provides wrappers for variants of RNN in Tensorflow
RNN变种
"""

import tensorflow as tf
import tensorflow.contrib as tc


def rnn(rnn_type, inputs, length, hidden_size, layer_num=1,
        dropout_keep_prob=None, concat=True):
    """
    Implements (Bi-)LSTM, (Bi-)GRU and (Bi-)RNN
    在这个module中，rnn是主要的接口，所以把rnn放在上面
    Args:
        rnn_type: the type of rnn, such as lstm
        inputs: padded inputs into rnn, usually a d*p or l*p matrix
        length: the valid length of the inputs,
        usually the length of the sentence
        hidden_size: the size of hidden units
        layer_num: multiple rnn layer are stacked if layer_num > 1
        dropout_keep_prob: dropout in RNN
        concat: When the rnn is bidirectional, the forward outputs and backward
        outputs are concatenated (such as a 2l*p matrix) if this is True,
        else we add them (add two matrices).
    Returns:
        RNN outputs and final state (such as the state of lstm)
    """
    if not rnn_type.startswith('bi'):
        cell = get_cell(rnn_type, hidden_size, layer_num, dropout_keep_prob)
        # 得到cell，在z轴、y轴已经展开，但是在x轴上并没有延展
        outputs, state = tf.nn.dynamic_rnn(cell, inputs,
                                           sequence_length=length,
                                           dtype=tf.float32)
        # 利用dynamic_rnn函数对cell在x轴方向上进行延展，并且把cell的inputs输入
        # outputs的维度是hidden_size*length, state的维度是hidden_size*layer_num*2

        if rnn_type.endswith('lstm'):
            c, h = state
            state = h
            # 把hidden state作为state
    else:  # bidirectional rnn
        cell_fw = get_cell(rnn_type, hidden_size, layer_num, dropout_keep_prob)
        # forward cell
        cell_bw = get_cell(rnn_type, hidden_size, layer_num, dropout_keep_prob)
        # backward cell
        outputs, state = tf.nn.bidirectional_dynamic_rnn(
            cell_bw, cell_fw, inputs, sequence_length=length, dtype=tf.float32
        )
        # 双向rnn相比单向rnn，在hidden_size这个维度上变成了之前的2倍

        state_fw, state_bw = state
        # 首先把state分离成forward state和backward state

        if rnn_type.endswith('lstm'):
            c_fw, h_fw = state_fw
            c_bw, h_bw = state_bw
            state_fw, state_bw = h_fw, h_bw
            # 对于lstm来说，我们要的state是hidden state
        if concat:
            outputs = tf.concat(outputs, 2)
            # 把两个tensor沿着hidden_size的维度连起来
            state = tf.concat([state_fw, state_bw], 1)
            # state同样要沿着hidden_size的维度连起来
        else:
            outputs = outputs[0] + outputs[1]
            state = state_fw + state_bw
            # 简单向量（张量）相加或者做平均处理
    return outputs, state
# 输出是两个张量，表示整个rnn网络的输出，有向上的输出outputs，也有向右的输出state


def get_cell(rnn_type, hidden_size, layer_num=1, dropout_keep_prob=None):
    """
    Get the RNN Cell
    Args:
        rnn_type: 'lstm', 'gru' or 'rnn'
        hidden_size: The size of hidden units
        layer_num: tc.rnn.MultiRNNCell are used if layer_num > 1
        dropout_keep_prob: dropout in RNN cell
    Returns:
        An RNN Cell
    """
    if rnn_type.endswith('lstm'):
        cell = tc.rnn.LSTMCell(num_units=hidden_size, state_is_tuple=True)
        # returns an instance
        # num_units is the z direction of the RNN network
        # state_is_tuple: If True, accepted and returned states are 2-tuples
        # of the c_state and m_state. If False, they are concatenated along
        # the column axis. This latter behavior will soon be deprecated.
        # https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/LSTMCell
        # 在lstm中，有ct，也有ht，在cell之间流动，同时ht在当步输出，最终汇总成output
        # c_state就是ct, m_state就是ht
    elif rnn_type.endswith('gru'):
        cell = tc.rnn.GRUCell(num_units=hidden_size)
    elif rnn_type.endswith('rnn'):
        cell = tc.rnn.BasicRNNCell(num_units=hidden_size)
    else:
        raise NotImplementedError('Unsuported rnn type: {}'.format(rnn_type))
    # 对输入参数的完备的考虑
    if dropout_keep_prob is not None:
        cell = tc.rnn.DropoutWrapper(cell,
                                     input_keep_prob=dropout_keep_prob,
                                     output_keep_prob=dropout_keep_prob)
        # 在z轴方向上的cell，每层cell后面都要接dropout，随机毙掉一定比例的cell
        # input_keep_prob: unit Tensor or float between 0 and 1,
        # input keep probability; if it is constant and 1,
        # no input dropout will be added. 进来的时候就要毙掉一定比例的cell
        # output_keep_prob: 出去的时候也要毙掉一定比例的cell
    if layer_num > 1:
        cell = tc.rnn.MultiRNNCell([cell]*layer_num, state_is_tuple=True)
        # 经过这个类处理后，在y轴方向上叠加多个cell
        # 如何添加多层, 把一个单层的cell变成多层的cell
        # 此处各层的hidden units数目是相同的，所以是[cell, cell, ...]
        # 也可以设置成不同的，这样列表里就是不同的cell了
    return cell

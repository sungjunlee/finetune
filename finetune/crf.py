import numpy as np
import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.layers import utils
from tensorflow.python.ops import rnn
from tensorflow.python.ops import variable_scope as vs
from tensorflow.contrib.crf import CrfDecodeBackwardRnnCell, CrfDecodeForwardRnnCell, CrfForwardRnnCell

from finetune.utils import np_softmax


def crf_sequence_score(inputs, tag_indices, sequence_lengths,
                       transition_params):
    """Computes the unnormalized score for a tag sequence.
    Args:
        inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
            to use as input to the CRF layer.
        tag_indices: A [batch_size, max_seq_len] matrix of tag indices for which we
            compute the unnormalized score.
        sequence_lengths: A [batch_size] vector of true sequence lengths.
        transition_params: A [num_tags, num_tags] transition matrix.
    Returns:
        sequence_scores: A [batch_size] vector of unnormalized sequence scores.
    """
    # If max_seq_len is 1, we skip the score calculation and simply gather the
    # unary potentials of the single tag.
    def _single_seq_fn():
        batch_size = array_ops.shape(inputs, out_type=tag_indices.dtype)[0]
        example_inds = array_ops.reshape(
            math_ops.range(batch_size, dtype=tag_indices.dtype), [-1, 1])
        sequence_scores = array_ops.gather_nd(
            array_ops.squeeze(inputs, [1]),
            array_ops.concat([example_inds, tag_indices], axis=1))
        sequence_scores = array_ops.where(
            math_ops.less_equal(sequence_lengths, 0),
            array_ops.zeros_like(sequence_scores),
            sequence_scores
        )
        return sequence_scores

    def _multi_seq_fn():
        # Compute the scores of the given tag sequence.
        unary_scores = crf_unary_score(tag_indices, sequence_lengths, inputs)
        binary_scores = crf_binary_score(tag_indices, sequence_lengths,
                                        transition_params)
        sequence_scores = unary_scores + tf.pad(binary_scores, tf.constant([[0, 0], [1, 0]]), "CONSTANT")
        # sequence_scores = math_ops.reduce_sum(unary_scores + binary_scores, 1)
        return sequence_scores

    return utils.smart_cond(
        pred=math_ops.equal(inputs.shape[1].value or array_ops.shape(inputs)[1], 1),
        true_fn=_single_seq_fn,
        false_fn=_multi_seq_fn
    )


def crf_log_norm(inputs, sequence_lengths, transition_params):
    """Computes the normalization for a CRF.
    Args:
        inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
            to use as input to the CRF layer.
        sequence_lengths: A [batch_size] vector of true sequence lengths.
        transition_params: A [num_tags, num_tags] transition matrix.
    Returns:
        log_norm: A [batch_size] vector of normalizers for a CRF.
    """
    # Split up the first and rest of the inputs in preparation for the forward
    # algorithm.
    first_input = array_ops.slice(inputs, [0, 0, 0], [-1, 1, -1])
    first_input = array_ops.squeeze(first_input, [1])

    # If max_seq_len is 1, we skip the algorithm and simply reduce_logsumexp over
    # the "initial state" (the unary potentials).
    def _single_seq_fn():
        log_norm = math_ops.reduce_logsumexp(first_input, [1])
        # Mask `log_norm` of the sequences with length <= zero.
        log_norm = array_ops.where(math_ops.less_equal(sequence_lengths, 0),
                                array_ops.zeros_like(log_norm),
                                log_norm)
        return log_norm

    def _multi_seq_fn():
        """Forward computation of alpha values."""
        rest_of_input = array_ops.slice(inputs, [0, 1, 0], [-1, -1, -1])

        # Compute the alpha values in the forward algorithm in order to get the
        # partition function.
        forward_cell = CrfForwardRnnCell(transition_params)
        # Sequence length is not allowed to be less than zero.
        sequence_lengths_less_one = math_ops.maximum(
            constant_op.constant(0, dtype=sequence_lengths.dtype),
            sequence_lengths - 1)
        _, alphas = rnn.dynamic_rnn(
            cell=forward_cell,
            inputs=rest_of_input,
            sequence_length=sequence_lengths_less_one,
            initial_state=first_input,
            dtype=dtypes.float32)
        log_norm = math_ops.reduce_logsumexp(alphas, [1])
        # Mask `log_norm` of the sequences with length <= zero.
        log_norm = array_ops.where(
            math_ops.less_equal(sequence_lengths, 0),
            array_ops.zeros_like(log_norm),
            log_norm
        )
        return log_norm

    max_seq_len = array_ops.shape(inputs)[1]
    return control_flow_ops.cond(pred=math_ops.equal(max_seq_len, 1),
                                true_fn=_single_seq_fn,
                                false_fn=_multi_seq_fn)


def crf_log_likelihood(inputs, tag_indices, sequence_lengths, transition_params=None, class_weights=None):
    """Computes the log-likelihood of tag sequences in a CRF.
    Args:
        inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
            to use as input to the CRF layer.
        tag_indices: A [batch_size, max_seq_len] matrix of tag indices for which we
            compute the log-likelihood.
        sequence_lengths: A [batch_size] vector of true sequence lengths.
        transition_params: A [num_tags, num_tags] transition matrix, if available.
    Returns:
        log_likelihood: A [batch_size] `Tensor` containing the log-likelihood of
        each example, given the sequence of tag indices.
        transition_params: A [num_tags, num_tags] transition matrix. This is either
            provided by the caller or created in this function.
    """
    # Get shape information.
    num_tags = inputs.get_shape()[2].value

    # Get the transition matrix if not provided.
    if transition_params is None:
        transition_params = vs.get_variable("transitions", [num_tags, num_tags])

    sequence_scores = crf_sequence_score(inputs, tag_indices, sequence_lengths, transition_params)

    if class_weights is not None:
        flat_loss = tf.reshape(sequence_scores, [-1])
        flat_targets = tf.reshape(tag_indices, [-1])
        one_hot_targets = tf.one_hot(flat_targets, depth=num_tags)

        # loss multiplier applied based on true class
        weights = tf.reduce_sum(class_weights * tf.to_float(one_hot_targets), axis=1)
        flat_loss *= weights
        sequence_scores = tf.reshape(flat_loss, tf.shape(sequence_scores))

    sequence_scores = tf.reduce_sum(sequence_scores, 1)
    log_norm = crf_log_norm(inputs, sequence_lengths, transition_params)

    # Normalize the scores to get the log-likelihood per example.
    log_likelihood = sequence_scores - log_norm
    return log_likelihood, transition_params


def crf_unary_score(tag_indices, sequence_lengths, inputs):
    """Computes the unary scores of tag sequences.
    Args:
        tag_indices: A [batch_size, max_seq_len] matrix of tag indices.
        sequence_lengths: A [batch_size] vector of true sequence lengths.
        inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials.
    Returns:
        unary_scores: A [batch_size] vector of unary scores.
    """
    batch_size = array_ops.shape(inputs)[0]
    max_seq_len = array_ops.shape(inputs)[1]
    num_tags = array_ops.shape(inputs)[2]

    flattened_inputs = array_ops.reshape(inputs, [-1])

    offsets = array_ops.expand_dims(
        math_ops.range(batch_size) * max_seq_len * num_tags, 1)
    offsets += array_ops.expand_dims(math_ops.range(max_seq_len) * num_tags, 0)
    # Use int32 or int64 based on tag_indices' dtype.
    if tag_indices.dtype == dtypes.int64:
        offsets = math_ops.to_int64(offsets)
    flattened_tag_indices = array_ops.reshape(offsets + tag_indices, [-1])

    unary_scores = array_ops.reshape(
        array_ops.gather(flattened_inputs, flattened_tag_indices),
        [batch_size, max_seq_len])

    masks = array_ops.sequence_mask(sequence_lengths,
                                    maxlen=array_ops.shape(tag_indices)[1],
                                    dtype=dtypes.float32)
    return unary_scores * masks


def crf_binary_score(tag_indices, sequence_lengths, transition_params):
    """Computes the binary scores of tag sequences.
    Args:
        tag_indices: A [batch_size, max_seq_len] matrix of tag indices.
        sequence_lengths: A [batch_size] vector of true sequence lengths.
        transition_params: A [num_tags, num_tags] matrix of binary potentials.
    Returns:
        binary_scores: A [batch_size] vector of binary scores.
    """
    # Get shape information.
    num_tags = transition_params.get_shape()[0]
    num_transitions = array_ops.shape(tag_indices)[1] - 1

    # Truncate by one on each side of the sequence to get the start and end
    # indices of each transition.
    start_tag_indices = array_ops.slice(tag_indices, [0, 0],
                                        [-1, num_transitions])
    end_tag_indices = array_ops.slice(tag_indices, [0, 1], [-1, num_transitions])

    # Encode the indices in a flattened representation.
    flattened_transition_indices = start_tag_indices * num_tags + end_tag_indices
    flattened_transition_params = array_ops.reshape(transition_params, [-1])

    # Get the binary scores based on the flattened representation.
    binary_scores = array_ops.gather(flattened_transition_params,
                                    flattened_transition_indices)

    masks = array_ops.sequence_mask(sequence_lengths,
                                    maxlen=array_ops.shape(tag_indices)[1],
                                    dtype=dtypes.float32)
    
    # model
    truncated_masks = array_ops.slice(masks, [0, 1], [-1, -1])
    binary_scores *= truncated_masks
    return binary_scores 


def viterbi_decode(score, transition_params):
    """Decode the highest scoring sequence of tags outside of TensorFlow.
    This should only be used at test time.
    Args:
        score: A [seq_len, num_tags] matrix of unary potentials.
        transition_params: A [num_tags, num_tags] matrix of binary potentials.
    Returns:
        viterbi: A [seq_len] list of integers containing the highest scoring tag
            indices.
        viterbi_score: A float containing the score for the Viterbi sequence.
    """
    trellis = np.zeros_like(score)
    backpointers = np.zeros_like(score, dtype=np.int32)
    trellis[0] = score[0]

    for t in range(1, score.shape[0]):
        v = np.expand_dims(trellis[t - 1], 1) + transition_params
        trellis[t] = score[t] + np.max(v, 0)
        backpointers[t] = np.argmax(v, 0)

    viterbi = [np.argmax(trellis[-1])]
    for bp in reversed(backpointers[1:]):
        viterbi.append(bp[viterbi[-1]])
    viterbi.reverse()

    return viterbi, np_softmax(trellis, axis=-1)


def sequence_decode(logits, transition_matrix):
    """ A simple py_func wrapper around the Viterbi decode allowing it to be included in the tensorflow graph. """

    def _sequence_decode(logits, transition_matrix):
        all_predictions = []
        all_logits = []
        for logit in logits:
            viterbi_sequence, viterbi_logits = viterbi_decode(logit, transition_matrix)
            all_predictions.append(viterbi_sequence)
            all_logits.append(viterbi_logits)
        return np.array(all_predictions, dtype=np.int32), np.array(all_logits, dtype=np.float32)

    return tf.py_func(_sequence_decode, [logits, transition_matrix], [tf.int32, tf.float32])

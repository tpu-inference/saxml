# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Wraps a model with LMService APIs."""

import abc
import dataclasses
import functools
import inspect
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from absl import logging
import jax
from jax import experimental as jax_exp
from jax import numpy as jnp
import numpy as np
import orbax.export as oex
from praxis import base_layer
from praxis import base_model
from praxis import decoder_hparams
from praxis import decoder_utils
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes
from praxis.layers import multi_query_attention
from saxml.server.jax import np_tf_sess_wrapper
from saxml.server.jax import servable_model as jax_servable_model
from saxml.server.pax import servable_model
from saxml.server.pax import servable_model_params
from saxml.server.pax.lm import lm_tokenizer
from saxml.server.pax.lm import servable_lm_common
from saxml.server.services import lm_service
import tensorflow as tf


JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor
NestedNpTensor = pytypes.NestedNpTensor
PRNGKey = pytypes.PRNGKey
NestedMap = py_utils.NestedMap
NestedPartitionSpec = pytypes.NestedPartitionSpec
NestedTfTensor = pytypes.Nested[tf.Tensor]
NestedNpOrTfTensor = Union[NestedNpTensor, NestedTfTensor]
LMMethodName = lm_service.LMMethodName
HostTensors = servable_model.HostTensors
ShapesAndDtypes = servable_model.ShapesAndDtypes
InputShapeInfo = servable_lm_common.InputShapeInfo
TensorSpec = servable_lm_common.TensorSpec
DeviceTensors = Any

decode_tf_post_processing = servable_lm_common.decode_tf_post_processing


@dataclasses.dataclass
class ScoreHParams(servable_model_params.ServableMethodParams):
  """HParameters for LM score method.

  Attributes:
    max_input_seq_len: static prefix sequence length dimension size.
    max_suffix_seq_len: static suffix sequence length dimension size. Defaults
      to be equal to `max_input_seq_len` if not set. Inputs are padded or
      truncated to (max_input_seq_len + max_suffix_seq_len) size.
    include_eos_score: whether to add EOS score to the result.
  """

  max_input_seq_len: int = 0
  max_suffix_seq_len: int = 0
  include_eos_score: bool = False
  fetch_prefix_lengths_from_inputs: bool = False
  output_geometric_mean_prob_score: bool = False


@dataclasses.dataclass
class DecodeHParams(servable_model_params.ServableMethodParams):
  """HParameters for LM sample decode method.

  Attributes:
    max_input_seq_len: static sequence length dimension size. Inputs are padded
      or truncated to this size.
    decoder: decoder params.
    include_prefix_in_result: whether to include the input prefix in the result.
    encoder_decoder_model: whether this is an encoder decoder model.
    t5_model: whether this is a T5 flaxformer based model.
    output_geometric_mean_prob_score: Whether to return geometric mean of prob
      score instead of sum of log prob as the score.
    output_avg_entropy_score: Whether to return avg entropy score instead of sum
      of log prob as the score.
  """

  max_input_seq_len: int = 0
  decoder: decoder_hparams.DecoderHParams = dataclasses.field(
      default_factory=decoder_hparams.DecoderHParams
  )
  include_prefix_in_result: bool = False
  encoder_decoder_model: bool = False
  t5_model: bool = False
  stream_interval_steps: int = 1
  fetch_prefix_lengths_from_inputs: bool = False
  output_geometric_mean_prob_score: bool = False
  output_avg_entropy_score: bool = False


@dataclasses.dataclass
class TextToEmbeddingHParams(servable_model_params.ServableMethodParams):
  """HParameters for TextToEmbedding method.

  Attributes:
    max_input_seq_len: static prefix sequence length dimension size.
    max_suffix_seq_len: static suffix sequence length dimension size. Defaults
      to 1 and `max_input_seq_len` is autodecremented by 1. This is to ensure
      the prefix and suffix both have EOS for tokenization. Inputs are padded or
      truncated to (max_input_seq_len + max_suffix_seq_len) size.
    include_eos_score: whether to add EOS score to the result.
    output_embedding_name: The name of the embedding to use from the model's
      outputs.  Required.
    output_padding_name: The name of padding to use from the model's outputs.
      This is used when output embedding has a variable length. For example,
      returning embeddings of all tokens in a sequence, rather than pooling one
      embedding out.
    model_method_name: The name of the method to call to extract embeddings from
      an input image.  Required.
  """

  max_input_seq_len: int = 0
  max_suffix_seq_len: int = 1
  include_eos_score: bool = False
  output_embedding_name: Optional[str] = None
  output_padding_name: Optional[str] = None
  model_method_name: Optional[str] = None


@dataclasses.dataclass
class GradientHParams(ScoreHParams):
  """HParameters for LM gradient method, inheriting from ScoreHParams.

  Additional attributes:
    inputs_tensor_names: tensors to take gradients with respect to in inputs.
    mdl_vars_tensors_names: tensors to take gradients with respect to in
      mdl_vars.
  """

  inputs_tensor_names: Optional[List[str]] = None
  mdl_vars_tensor_names: Optional[List[str]] = None


class ServableLMModelParams(
    servable_model_params.ServableModelParams, metaclass=abc.ABCMeta
):
  """A base class that each LM model config needs to implement for serving."""

  @abc.abstractmethod
  def serving_tokenizer(self) -> pax_fiddle.Config[lm_tokenizer.LMTokenizer]:
    """Tokenizer params used by serving."""

  def methods(self) -> Dict[str, servable_model_params.ServableMethodParams]:
    methods = {}
    score = self.score()  # pylint: disable=assignment-from-none
    if score is not None:
      methods[LMMethodName.SCORE] = score
    generate = self.generate()  # pylint: disable=assignment-from-none
    if generate is not None:
      methods[LMMethodName.GENERATE] = generate
    generate_stream = self.generate_stream()  # pylint: disable=assignment-from-none
    if generate_stream is not None:
      methods[LMMethodName.GENERATE_STREAM] = generate_stream
    text_to_embedding = self.text_to_embedding()  # pylint: disable=assignment-from-none
    if text_to_embedding is not None:
      methods[LMMethodName.EMBED] = text_to_embedding
    gradient = self.gradient()  # pylint: disable=assignment-from-none
    if gradient is not None:
      methods[LMMethodName.GRADIENT] = gradient
    return methods

  def score(self) -> Optional[ScoreHParams]:
    """Returns the params for the score method."""
    return None

  def generate(self) -> Optional[DecodeHParams]:
    """Returns the params for the decode method."""
    return None

  def gradient(self) -> Optional[GradientHParams]:
    """Returns the params for the gradient method."""
    return None

  def generate_stream(self) -> Optional[DecodeHParams]:
    """Returns the params for the decode method."""
    return None

  def text_to_embedding(self) -> Optional[TextToEmbeddingHParams]:
    return None

  def create_model(self, primary_process_id: int) -> 'ServableLMModel':
    return ServableLMModel(
        self,
        primary_process_id,
        self.get_checkpoint_type(),
        test_mode=self.test_mode,
        enable_auto_sharding=self.enable_auto_sharding,
        compiler_options=self.compiler_options(),
        do_eval=self.do_eval,
    )


class ServableLMMethod(servable_model.ServableMethod):
  """Implements common method of LM."""

  @classmethod
  def service_id(cls) -> str:
    return lm_service.SERVICE_ID

  @property
  def sorted_seq_lens(self) -> List[int]:
    """A list of sorted supported (ascending order) sequence lengths."""
    return sorted(self._bucket_keys) if self._bucket_keys else [-1]

  def get_sorted_input_shapes(self) -> List[InputShapeInfo]:
    result = []
    for batch_size in self._sorted_batch_sizes:
      for seq_len in self.sorted_seq_lens:
        result.append(InputShapeInfo(batch_size, seq_len))
    return result

  def deserialize_input_shape(self, unpadded_shape_str: str) -> InputShapeInfo:
    """Deserialize input shape from a str."""
    return servable_lm_common.deserialize_input_shape(
        unpadded_shape_str, self._dummy_bucket_key
    )

  def get_unpadded_shape(
      self, unpadded_batch_size, inputs: HostTensors
  ) -> InputShapeInfo:
    return InputShapeInfo(
        unpadded_batch_size,
        servable_lm_common.get_max_seq_len_in_batch(
            inputs, self._dummy_bucket_key, self._bucket_keys
        ),
    )

  def get_padded_input_shape(
      self, unpadded_shape: InputShapeInfo
  ) -> InputShapeInfo:
    """Get padded input shape.

    Args:
      unpadded_shape: Unpadded shape information contains batch size or sequence
        length.

    Returns:
      Padded input shape.
    Raises:
      ValueError if unpadded batch size or sequence length too large.
    """
    padded_shape = super().get_padded_input_shape(unpadded_shape)
    if self._bucket_keys is None:
      return InputShapeInfo(padded_shape.batch_size)
    padded_seq_len = servable_lm_common.get_padded_input_seq_len(
        unpadded_shape.seq_len, self.sorted_seq_lens
    )
    return InputShapeInfo(padded_shape.batch_size, padded_seq_len)

  def get_dummy_inputs(self, input_shape: InputShapeInfo) -> HostTensors:
    """Returns host tensors with dummy data at a batch size."""
    batched_input = self.pre_processing(
        [self._dummy_input_sample] * input_shape.batch_size
    )

    return servable_lm_common.handle_host_input_with_input_shape(
        batched_input, input_shape
    )

  def resize_host_array(
      self,
      x: np.ndarray,
      global_input_shape_dtype: ShapesAndDtypes,
      unpadded_input_shape: InputShapeInfo,
  ):
    """Resizes x to the desired shape.

    Args:
      x: Host tensor.
      global_input_shape_dtype: Global input shape and dtype for this tensor.
      unpadded_input_shape: Unpadded input shape.

    Returns:
      host array after padding or slice of x.
    """
    x = servable_lm_common.resize_host_array(
        x, global_input_shape_dtype, unpadded_input_shape
    )

    # Let the parent class handle the batch dim.
    x = super().resize_host_array(
        x, global_input_shape_dtype, unpadded_input_shape
    )
    return x

  def _get_longest_seqlen(self, inputs: NestedNpTensor) -> int:
    """Gets the longest sequence length in a batch."""
    if 'paddings' in inputs:
      prefix_lengths = np.sum(1.0 - inputs['paddings'], axis=-1).astype(
          np.int32
      )  # pytype: disable=attribute-error
      return np.max(prefix_lengths).item()
    return inputs['ids'].shape[1]

  def get_unpadded_branch_key(self, inputs: NestedNpTensor) -> int:
    return self._get_longest_seqlen(inputs)

  def get_branch_inputs(
      self, inputs: NestedJTensor, branch_key: int
  ) -> NestedJTensor:
    """Returns the inputs for a branch key.

    Args:
      inputs: inputs with padded sequence lengths.
      branch_key: branch_key is seqlen.

    Returns:
      Tensors sliced at sequence length dimension.
    """
    seqlen = branch_key

    def _slice_fn(x):
      """The function to slice at sequence dimension."""
      if not isinstance(x, JTensor):
        return x
      if len(x.shape) == 2 and x.shape[1] >= seqlen:
        return jax.lax.slice(x, [0, 0], [x.shape[0], seqlen])
      return x

    return jax.tree_util.tree_map(_slice_fn, inputs)

  def get_maxlen(self) -> int:
    """Gets the max input sequence lengths."""
    raise NotImplementedError('get_maxlen not implemented')

  def output_seq_dim(self) -> int:
    """Gets the sequence dim in the output result."""
    raise NotImplementedError('output_seq_dim not implemented')

  def extra_pad_result(
      self, result: NestedJTensor, branch_key: int
  ) -> NestedJTensor:
    """Special paddings for some tensors."""
    return result

  def pad_result(
      self, result: NestedJTensor, pad_len: int, seq_dim: int
  ) -> NestedJTensor:
    """Pads the result at sequence dimension."""

    def _pad_fn(x):
      if not isinstance(x, JTensor) or len(x.shape) < seq_dim + 1:
        return x
      paddings = [[0, 0]] * len(x.shape)
      paddings[seq_dim] = [0, max(0, pad_len)]
      padded = jnp.pad(x, paddings)
      return padded

    return jax.tree_map(_pad_fn, result)

  def post_process_branch_outputs(
      self, outputs: NestedJTensor, branch_key: int
  ) -> NestedJTensor:
    """Post process branch outputs."""
    seqlen = branch_key
    maxlen = self.get_maxlen()
    result, state = outputs
    padded_result = self.pad_result(
        result, maxlen - seqlen, self.output_seq_dim()
    )
    padded_result = self.extra_pad_result(padded_result, branch_key)
    padded_state = self.pad_result(state, maxlen - seqlen, 1)
    return padded_result, padded_state

  @property
  def model_fn_input_polymorphic_shape(self) -> pytypes.Nested[str]:
    """Returns a batch polymorphic shape for jax2tf."""
    batched_host_dummy = self.get_dummy_inputs(InputShapeInfo(self.batch_size))
    batched_host_dummy = self.update_extra_inputs(
        batched_host_dummy,
        self.batch_size,
        [self.default_extra_inputs] * self.batch_size,
    )

    batch_pattern = 'b' if len(self.sorted_batch_sizes) > 1 else '_'
    if len(self.sorted_seq_lens) > 1:
      seq_pattern = f'{batch_pattern}, t'
    else:
      seq_pattern = f'{batch_pattern}, _'
    shape_patterns = jax.tree_util.tree_map(
        lambda x: seq_pattern if len(x.shape) == 2 else f'{batch_pattern}, ...',
        batched_host_dummy,
    )
    # Apply seq len polymorphism exclusion.
    polymorphic_seq_len_exclusion = set(
        self.method_params.polymorphic_seq_len_exclusion or []
    )
    # Do not apply polymorphic seq len to extra inputs.
    if self.default_extra_inputs:
      polymorphic_seq_len_exclusion |= self.default_extra_inputs.keys()
    if polymorphic_seq_len_exclusion:
      for key in polymorphic_seq_len_exclusion:
        shape_patterns[key] = f'{batch_pattern}, ...'

    return shape_patterns


class LMScoreMethod(ServableLMMethod):
  """Implements the score method of an LM."""

  def __init__(
      self,
      model: base_model.BaseModel,
      model_state: servable_model.ServableModelState,
      prng_key: PRNGKey,
      score_params: ScoreHParams,
      tokenizer_p: Any,
      exportable: bool = False,
      enable_auto_sharding: bool = False,
      compiler_options: jax.stages.CompilerOptions | None = None,
  ):
    self._tokenizer = tokenizer_p.Instantiate()
    self._score_params = score_params
    dummy_input_sample = ('', [''])
    # TODO(b/289379065): Remove this workaround here.
    if (
        isinstance(self._tokenizer, lm_tokenizer.LMTokenizer)
        and self._tokenizer.tokenized_input
    ):
      dummy_input_sample = ('1', ['1'])
    logging.info('Using np_tf_sess_wrapper on LMScoreMethod.tf_pre_processing')
    self._tf_sess_pre_processing = np_tf_sess_wrapper.wrap_tf_session(
        # `bucketize_inputs` is only used in SavedModel export. The sax-native
        # serving has an equivalent bucketization after `pre_processing`.
        lambda *args: self.tf_pre_processing(*args, bucketize_inputs=False)
    )
    super().__init__(
        model,
        'compute_predictions',
        model_state,
        score_params,
        prng_key,
        dummy_input_sample,
        exportable=exportable,
        enable_auto_sharding=enable_auto_sharding,
        compiler_options=compiler_options,
    )

  def fetch_output(
      self, model_fn_outputs: NestedJTensor, model_fn_inputs: NestedJTensor
  ) -> NestedJTensor:
    if 'scores' in model_fn_outputs[0]:
      # Custom scores.
      return model_fn_outputs[0]['scores']
    # per_token_xent or per_example_xnent is -logprobs. We return the negative
    # value so that higher score is better.
    if 'per_token_xent' not in model_fn_outputs[0]:
      assert 'per_example_xent' in model_fn_outputs[0]
      assert model_fn_outputs[0].per_example_xent.ndim == 1  # pytype: disable=attribute-error  # jax-ndarray
      return -model_fn_outputs[0].per_example_xent  # pytype: disable=attribute-error  # jax-ndarray
    assert len(model_fn_outputs[0].per_token_xent.shape) > 1  # pytype: disable=attribute-error  # jax-ndarray
    xnent_len = model_fn_outputs[0].per_token_xent.shape[1]  # pytype: disable=attribute-error  # jax-ndarray
    assert xnent_len == model_fn_inputs.ids.shape[1]  # pytype: disable=attribute-error  # jax-ndarray
    per_token_logprobs = -model_fn_outputs[0].per_token_xent  # pytype: disable=attribute-error  # jax-ndarray
    non_paddings = 1.0 - model_fn_inputs.paddings  # pytype: disable=attribute-error  # jax-ndarray
    if not self._score_params.include_eos_score and self._tokenizer.append_eos:
      non_paddings = jnp.pad(
          # TODO(b/263808957): change back to non_paddings[:, 1:] once the bug
          # is fixed.
          jax.lax.dynamic_slice_in_dim(
              non_paddings, 1, non_paddings.shape[1] - 1, axis=1
          ),
          [[0, 0], [0, 1]],
      )
    sum_per_token_logprobs = jnp.sum(
        per_token_logprobs * model_fn_inputs.score_masks * non_paddings,  # pytype: disable=attribute-error  # jax-ndarray
        axis=-1,
        keepdims=True,
    )
    if self._score_params.output_geometric_mean_prob_score:
      num_output_tokens = jnp.sum(
          model_fn_inputs.score_masks * non_paddings,  # pytype: disable=attribute-error  # jax-ndarray
          axis=-1,
          keepdims=True,
      )
      num_output_tokens = jnp.where(num_output_tokens > 0, num_output_tokens, 1)
      return jnp.exp(sum_per_token_logprobs / num_output_tokens)
    else:
      return sum_per_token_logprobs

  def get_maxlen(self) -> int:
    return (
        self._score_params.max_input_seq_len
        + self._score_params.max_suffix_seq_len
    )

  def output_seq_dim(self) -> int:
    return 1

  def pre_processing(
      self, raw_inputs: List[Tuple[str, List[str]]]
  ) -> NestedNpTensor:
    prefixes = np.array([prefix for prefix, _ in raw_inputs])
    for _, suffix in raw_inputs:
      assert len(suffix) <= 1, 'Only one suffix score is supported in lm.score'
    suffixes = np.array([suffix[0] for _, suffix in raw_inputs])

    # HuggingFace tokenizer based custom vocabularies are enabled by applying
    # tf.py_function. The preprocessing and postprocessing are wrapped by
    # np_tf_sess_wrapper.wrap_tf_session function to allow export SavedModel.
    # However, the np_tf_sess_wrapper.wrap_tf_session function does not know how
    # to handle tf.py_function when trying to create a SavedModel-exportable
    # GraphDef object. Thus, to use custom vocabularies, we skip applying
    # np_tf_sess_wrapper.wrap_tf_session to preprocessing and postprocessing.
    if (
        isinstance(self._tokenizer, lm_tokenizer.LMTokenizer)
        and self._tokenizer.vocabulary_class
        and not self._tokenizer.tokenized_input
    ):
      tf_pre_processed = self.tf_pre_processing(prefixes, suffixes)
      return jax.tree_util.tree_map(np.array, tf_pre_processed)
    return self._tf_sess_pre_processing(prefixes, suffixes)

  def post_processing(self, compute_outputs: NestedNpTensor) -> List[float]:
    assert isinstance(compute_outputs, pytypes.NpTensor)
    scores = list(compute_outputs.astype(float))
    return scores

  def tf_pre_processing(
      self,
      prefixes: NestedNpOrTfTensor,
      suffixes: NestedNpOrTfTensor,
      extra_inputs: Mapping[str, Any] | None = None,
      branch_index: NestedNpOrTfTensor | None = None,
      bucketize_inputs: bool = True,
  ) -> NestedTfTensor:
    """Tokenizes `prefixes` and `suffixes` using TF ops.

    This also implements `ExportableToSavedModel.tf_pre_processing`.

    Args:
      prefixes: the prefix text batch of shape [batch_size].
      suffixes: the suffix text batch of shape [batch_size].
      extra_inputs: optional mapping of extra input key to tensor or tensor spec
        of shape [batch_size].
      branch_index: optional index to indicate which bucket key will be used by
        `bucketize_tokenized_inputs`.
      bucketize_inputs: whether to bucketize the preprocessed inputs based on
        max sequence length in the batch.

    Returns:
      A NestedMap of preprocessed tensors.
    """
    preprocessed = servable_lm_common.tf_tokenize_inputs(
        prefixes,
        suffixes,
        self._tokenizer,
        self._score_params.max_input_seq_len,
        self._score_params.max_suffix_seq_len,
        self._score_params.include_eos_score,
    )

    if bucketize_inputs:
      preprocessed = servable_lm_common.bucketize_tokenized_inputs(
          self.sorted_seq_lens, preprocessed, branch_index
      )

    if extra_inputs:
      preprocessed.update(extra_inputs)

    return preprocessed

  def tf_post_processing(
      self, compute_outputs: NestedNpOrTfTensor
  ) -> NestedNpOrTfTensor:
    """Implements `ExportableToSavedModel.tf_post_processing`."""
    return {'scores': compute_outputs}

  def input_signature(
      self, batch_size: Optional[int]
  ) -> Tuple[TensorSpec, TensorSpec, Mapping[str, TensorSpec], TensorSpec]:
    """Implements `ExportableToSavedModel.input_signature`."""
    return (
        tf.TensorSpec([batch_size], dtype=tf.string, name='prefixes'),
        tf.TensorSpec([batch_size], dtype=tf.string, name='suffixes'),
        servable_lm_common.extra_inputs_to_tf_signature(
            self._extra_inputs,
            batch_size,
            self.method_params.extra_inputs_dtypes,
        ),
        oex.TensorSpecWithDefault(
            tf.TensorSpec(
                [batch_size], dtype=tf.int32, name='branch_index_warmup_only'
            ),
            tf.constant([-1], dtype=tf.int32, shape=[batch_size or 1]),
        ),
    )

  @property
  def tf_trackable_resources(self) -> Any:
    """Implements `ExportableToSavedModel.tf_trackable_resources`."""
    return None


class LMDecodeMethod(ServableLMMethod):
  """Base decode method of an LM."""

  def __init__(
      self,
      model: base_model.BaseModel,
      model_state: servable_model.ServableModelState,
      prng_key: PRNGKey,
      method_hparams: DecodeHParams,
      tokenizer_p: Any,
      exportable: bool = False,
      streamable_output: bool = False,
      load: bool = True,
      enable_auto_sharding: bool = False,
      compiler_options: jax.stages.CompilerOptions | None = None,
  ):
    self._tokenizer = tokenizer_p.Instantiate()
    self._method_hparams = method_hparams
    dummy_input_sample = ''
    # TODO(b/289379065): Remove this workaround here.
    if (
        isinstance(self._tokenizer, lm_tokenizer.LMTokenizer)
        and self._tokenizer.tokenized_input
    ):
      dummy_input_sample = '1'
    if isinstance(method_hparams, DecodeHParams):
      self._include_prefix_in_result = method_hparams.include_prefix_in_result
    logging.info('Using np_tf_sess_wrapper on LMDecodeMethod.tf_pre_processing')
    self._tf_sess_pre_processing = np_tf_sess_wrapper.wrap_tf_session(
        # `bucketize_inputs` is only used in SavedModel export. The sax-native
        # serving has an equivalent bucketization after `pre_processing`.
        lambda *args: self.tf_pre_processing(*args, bucketize_inputs=False)
    )
    logging.info(
        'Using np_tf_sess_wrapper on LMDecodeMethod.tf_post_processing'
    )
    self._tf_sess_post_processing = np_tf_sess_wrapper.wrap_tf_session(
        self.tf_post_processing,
        False,
    )
    self._streamable_output = streamable_output
    logging.info(
        'Initialize LMDecodeMethod to be streamable_output=%s.',
        streamable_output,
    )

    def _init_stream_and_decode(new_ids):
      batch_size = tf.shape(new_ids)[:-1]
      return self._tokenizer.DecodeOnStream(
          new_ids, self._tokenizer.InitStream(batch_size)
      )

    self._tf_sess_first_stream_step = np_tf_sess_wrapper.wrap_tf_session(
        _init_stream_and_decode, False
    )
    self._tf_sess_stream_step = np_tf_sess_wrapper.wrap_tf_session(
        self._tokenizer.DecodeOnStream, False
    )
    self._tf_sess_stream_finish = np_tf_sess_wrapper.wrap_tf_session(
        self._tokenizer.FinishStream, False
    )

    super().__init__(
        model,
        'decode',
        model_state,
        method_hparams,
        prng_key,
        dummy_input_sample,
        exportable=exportable,
        load=load,
        enable_auto_sharding=enable_auto_sharding,
        compiler_options=compiler_options,
    )

  def call_model_function(self, inputs, mdl_vars, prng_key):
    k1, k2 = prng_key

    kwargs = {}
    if self.streamable_output:

      def callback_fn(x):
        assert self.model_state.is_primary_host
        self.enqueue_stream_output(x)

      host_callback = functools.partial(
          jax_exp.io_callback,
          callback_fn,
          None,
          ordered=True,
          sharding=jax.sharding.SingleDeviceSharding(self.callback_device),
      )
      kwargs['result_callback'] = decoder_utils.StreamingResultCallback(
          host_callback,
          interval_steps=self._method_hparams.stream_interval_steps,
      )

    if (
        'callback_device_index'
        in inspect.signature(self._model.decode_with_params).parameters
    ):
      kwargs['callback_device_index'] = self.callback_device_index
    if (
        'callback_device'
        in inspect.signature(self._model.decode_with_params).parameters
    ):
      kwargs['callback_device'] = self.callback_device

    outputs = self._model.apply(
        mdl_vars,
        input_batch=inputs,
        method=self._model.decode_with_params,
        mutable=[
            base_layer.NON_TRAINABLE,
            base_layer.DECODE_CACHE,
            base_layer.PREFIX_DECODE_CACHE,
            base_layer.INTERMEDIATES,
        ],
        rngs={
            base_layer.PARAMS: k1,
            base_layer.RANDOM: k2,
        },
        decoder_params=self._method_hparams.decoder,
        **kwargs,
    )
    return outputs

  @property
  def streamable_output(self) -> bool:
    return self._streamable_output

  def fetch_output(
      self, model_fn_outputs: NestedJTensor, model_fn_inputs: NestedJTensor
  ) -> NestedJTensor:
    return servable_lm_common.decode_fetch_output(
        model_fn_outputs,
        model_fn_inputs,
        self._method_hparams.t5_model,
        self._method_hparams.fetch_prefix_lengths_from_inputs,
    )

  def pre_processing(self, raw_inputs: List[str]) -> NestedNpTensor:
    texts = np.array(raw_inputs)

    # HuggingFace tokenizer based custom vocabularies are enabled by applying
    # tf.py_function. The preprocessing and postprocessing are wrapped by
    # np_tf_sess_wrapper.wrap_tf_session function to allow export SavedModel.
    # However, the np_tf_sess_wrapper.wrap_tf_session function does not know how
    # to handle tf.py_function when trying to create a SavedModel-exportable
    # GraphDef object. Thus, to use custom vocabularies, we skip applying
    # np_tf_sess_wrapper.wrap_tf_session to preprocessing and postprocessing.
    if (
        isinstance(self._tokenizer, lm_tokenizer.LMTokenizer)
        and self._tokenizer.vocabulary_class
        and not self._tokenizer.tokenized_input
    ):
      tf_pre_processed = self.tf_pre_processing(texts)
      return jax.tree_util.tree_map(np.array, tf_pre_processed)
    return self._tf_sess_pre_processing(texts)

  def get_maxlen(self) -> int:
    return self._method_hparams.max_input_seq_len

  def output_seq_dim(self) -> int:
    return 2

  def extra_pad_result(
      self, result: NestedJTensor, branch_key: int
  ) -> NestedJTensor:
    """Extra pad result from decoding."""
    seqlen = branch_key

    def _pad_fn(sub_result):
      paddings = [[0, 0], [0, self.get_maxlen() - seqlen]]
      for key in {'paddings', 'weights', 'ids'}:
        if key in sub_result:
          sub_result[key] = jnp.pad(sub_result[key], paddings)
      return sub_result

    return tuple([_pad_fn(sub_result) for sub_result in result])

  def post_processing(
      self, compute_outputs: NestedNpTensor
  ) -> List[Tuple[List[str], List[float]]]:
    # A list of results for the inputs. Each element has multiple samples from
    # the decoding algorithm, which has a list of strings and a list of scores.

    # HuggingFace tokenizer based custom vocabularies are enabled by applying
    # tf.py_function. The preprocessing and postprocessing are wrapped by
    # np_tf_sess_wrapper.wrap_tf_session function to allow export SavedModel.
    # However, the np_tf_sess_wrapper.wrap_tf_session function does not know how
    # to handle tf.py_function when trying to create a SavedModel-exportable
    # GraphDef object. Thus, to use custom vocabularies, we skip applying
    # np_tf_sess_wrapper.wrap_tf_session to preprocessing and postprocessing.
    if (
        isinstance(self._tokenizer, lm_tokenizer.LMTokenizer)
        and self._tokenizer.vocabulary_class
        and not self._tokenizer.tokenized_output
    ):
      tf_post_processed = self.tf_post_processing(compute_outputs)
      post_processed = jax.tree_util.tree_map(np.array, tf_post_processed)
    else:
      post_processed = self._tf_sess_post_processing(compute_outputs)
    batched_decoded = post_processed['topk_decoded']
    batched_scores = post_processed['topk_scores']

    # Override scores according to hparams
    assert (
        not self._method_hparams.output_geometric_mean_prob_score
        or not self._method_hparams.output_avg_entropy_score
    )
    if self._method_hparams.output_geometric_mean_prob_score:
      num_output_tokens = np.count_nonzero(post_processed['topk_ids'], axis=2)
      num_output_tokens = np.where(num_output_tokens > 0, num_output_tokens, 1)
      batched_scores = np.exp(batched_scores / num_output_tokens)
    elif self._method_hparams.output_avg_entropy_score:
      batched_scores = post_processed['mean_entropy']

    # HuggingFace tokenizer based custom vocabularies are enabled by applying
    # tf.py_function. The preprocessing and postprocessing are wrapped by
    # np_tf_sess_wrapper.wrap_tf_session function to allow export SavedModel.
    # However, the np_tf_sess_wrapper.wrap_tf_session function does not know how
    # to handle tf.py_function when trying to create a SavedModel-exportable
    # GraphDef object. Thus, to use custom vocabularies, we skip applying
    # np_tf_sess_wrapper.wrap_tf_session to preprocessing and postprocessing.
    if (
        isinstance(self._tokenizer, lm_tokenizer.LMTokenizer)
        and self._tokenizer.vocabulary_class
        and not self._tokenizer.tokenized_output
    ):
      return [
          (decoded, list(scores))
          for decoded, scores in zip(batched_decoded, batched_scores)
      ]
    return [
        ([d.decode() for d in decoded], list(scores))
        for decoded, scores in zip(batched_decoded, batched_scores)
    ]

  def post_processing_stream(
      self,
      compute_outputs: Optional[NestedNpTensor] = None,
      stream_state: Optional[Any] = None,
  ) -> Tuple[List[Tuple[List[str], List[float]]], Optional[Any]]:
    if compute_outputs is None and stream_state is None:
      raise ValueError('compute_outputs and stream_state cannot both be None')

    if compute_outputs is None:
      batch_decoded = self._tf_sess_stream_finish(stream_state)
      stream_state = None
      scores = np.zeros(batch_decoded.shape)
    elif stream_state is None:
      batch_decoded, stream_state = self._tf_sess_first_stream_step(
          compute_outputs['output_ids']
      )
      scores = compute_outputs['scores']
    else:
      batch_decoded, stream_state = self._tf_sess_stream_step(
          compute_outputs['output_ids'], stream_state
      )
      scores = compute_outputs['scores']

    return [(d, s) for (d, s) in zip(batch_decoded, scores)], stream_state

  def tf_pre_processing(
      self,
      texts: NestedNpOrTfTensor,
      extra_inputs: Mapping[str, Any] | None = None,
      branch_index: NestedNpOrTfTensor | None = None,
      bucketize_inputs: bool = True,
  ) -> NestedTfTensor:
    """Tokenizes `texts` using TF ops.

    This also implements `ExportableToSavedModel.tf_pre_processing`. If extra
    inputs are provided in the input signature, the exported
    method will take a batched tensor too. See also the `input_signature` method
    of this class.

    Args:
      texts: the input text of shape [batch_size].
      extra_inputs: optional mapping of extra input key to tensor or tensor spec
        of shape [batch_size].
      branch_index: optional index to indicate which bucket key will be used by
        `bucketize_tokenized_inputs`.
      bucketize_inputs: whether to bucketize the preprocessed inputs based on
        max sequence length in the batch.

    Returns:
      A NestedMap of preprocessed tensors.
    """
    ids, paddings, prefix_lengths, weights = (
        servable_lm_common.decode_tf_tokenize_inputs(
            texts,
            self._tokenizer,
            self._method_hparams.max_input_seq_len,
            self._method_hparams.t5_model,
        )
    )

    batch_size = tf.shape(ids)[0]
    if self._method_hparams.t5_model:
      target_length = self._method_hparams.decoder.seqlen
      preprocessed = py_utils.NestedMap(
          encoder_input_tokens=ids,
          decoder_input_tokens=tf.ones((batch_size, target_length)),
      )
    elif self._method_hparams.encoder_decoder_model:
      src = py_utils.NestedMap(
          ids=tf.cast(ids, tf.int32),
          paddings=paddings,
      )
      tgt = py_utils.NestedMap(
          ids=tf.zeros((batch_size, 1), dtype=tf.int32),
          paddings=tf.zeros((batch_size, 1)),
      )
      preprocessed = py_utils.NestedMap(
          src=src,
          tgt=tgt,
          prefix_lengths=tf.ones((batch_size), tf.int32),
      )
    else:
      preprocessed = py_utils.NestedMap(
          ids=ids,
          paddings=paddings,
          prefix_lengths=tf.cast(prefix_lengths, tf.int32),
          weights=weights,
      )

    if bucketize_inputs:
      preprocessed = servable_lm_common.bucketize_tokenized_inputs(
          self.sorted_seq_lens,
          preprocessed,
          branch_index,
      )

    if extra_inputs:
      preprocessed.update(extra_inputs)

    return preprocessed

  def tf_post_processing(
      self, compute_outputs: NestedNpOrTfTensor
  ) -> NestedNpOrTfTensor:
    """Post-process the outputs using TF ops.

    This also implements `ExportableToSavedModel.tf_post_processing`.

    Args:
      compute_outputs: the outputs of the model function.

    Returns:
      A mapping that contains the decoded tensors, scores and ids of the topk
      results.
    """
    return decode_tf_post_processing(
        compute_outputs,
        tokenizer=self._tokenizer,
        t5_model=self._method_hparams.t5_model,
        include_prefix_in_result=self._include_prefix_in_result,
    )

  def input_signature(
      self, batch_size: Optional[int]
  ) -> Tuple[TensorSpec, Mapping[str, TensorSpec], TensorSpec]:
    """Implements `ExportableToSavedModel.input_signature`."""
    return (
        tf.TensorSpec([batch_size], dtype=tf.string, name='text'),
        servable_lm_common.extra_inputs_to_tf_signature(
            self._extra_inputs,
            batch_size,
            self.method_params.extra_inputs_dtypes,
        ),
        oex.TensorSpecWithDefault(
            tf.TensorSpec(
                [batch_size], dtype=tf.int32, name='branch_index_warmup_only'
            ),
            tf.constant([-1], dtype=tf.int32, shape=[batch_size or 1]),
        ),
    )

  @property
  def tf_trackable_resources(self) -> Any:
    """Implements `ExportableToSavedModel.tf_trackable_resources`."""
    return None


class LMDecodeMethodContinuousBatching(LMDecodeMethod):
  """Decode method support continuous batching."""

  def __init__(
      self,
      model: base_model.BaseModel,
      model_state: servable_model.ServableModelState,
      prng_key: PRNGKey,
      method_hparams: DecodeHParams,
      tokenizer_p: Any,
      exportable: bool = False,
      streamable_output: bool = False,
      load: bool = True,
      enable_auto_sharding: bool = False,
      compiler_options: jax.stages.CompilerOptions | None = None,
  ):
    self.decode_cache = NestedMap()
    self.decode_cache_pspecs = None

    self.decode_state = NestedMap()
    self.decode_state_pspecs = None

    super().__init__(
        model,
        model_state,
        prng_key,
        method_hparams,
        tokenizer_p,
        exportable,
        streamable_output,
        load,
        enable_auto_sharding,
        compiler_options,
    )

  @property
  def continuous_batching(self) -> bool:
    """Returns if the model method supports continuous batching."""
    return True

  @property
  def num_cache_slots(self) -> int:
    assert isinstance(self._method_params, DecodeHParams)
    return self._method_params.decoder.num_cache_slots

  @property
  def max_decode_steps(self) -> int:
    assert isinstance(self._method_params, DecodeHParams)
    max_decode_steps = self._method_params.decoder.max_decode_steps
    if isinstance(max_decode_steps, int):
      max_decode_steps = [max_decode_steps]
    return max(max_decode_steps)

  @property
  def input_sequence_len(self) -> int:
    assert isinstance(self._method_params, DecodeHParams)
    return self._method_params.decoder.seqlen - self.max_decode_steps

  @property
  def model_num_layers(self) -> int:
    return self._model.lm_tpl.stacked_transformer_tpl.num_layers

  @property
  def dim_per_head(self) -> int:
    return self._model.lm_tpl.stacked_transformer_tpl.dim_per_head

  @property
  def num_kv_heads(self) -> int:
    tr_atten_tpl = (
        self._model.lm_tpl.stacked_transformer_tpl.transformer_layer_params_tpl.tr_atten_tpl
    )
    if issubclass(
        tr_atten_tpl.cls, multi_query_attention.MultiQueryDotProductAttention
    ):
      return tr_atten_tpl.num_kv_heads
    return self._model.lm_tpl.stacked_transformer_tpl.num_heads

  def init_decode_cache(self):
    # decode cache mdl_vars
    num_layers = self.model_num_layers
    head_dims = self.dim_per_head
    sql_len = self.input_sequence_len + self.max_decode_steps
    consolidate_rope_key_state = False
    tr_atten_tpl = (
        self._model.lm_tpl.stacked_transformer_tpl.transformer_layer_params_tpl.tr_atten_tpl
    )
    if hasattr(tr_atten_tpl, 'consolidate_rope_key_state'):
      consolidate_rope_key_state = tr_atten_tpl.consolidate_rope_key_state

    def _init_decode_cache_fn():
      kv_cache = {}
      for i in range(num_layers):
        if self.num_kv_heads == 1:
          kv_state_shape = (self.num_cache_slots, sql_len, head_dims)
        else:
          kv_state_shape = (
              self.num_cache_slots,
              sql_len,
              self.num_kv_heads,
              head_dims,
          )
        layer_kv_cache = {
            'x_layers_{}'.format(i): {
                'self_attention': {
                    'key_state': jnp.zeros(
                        kv_state_shape,
                        dtype=self._model.fprop_dtype,
                    ),
                    'value_state': jnp.zeros(
                        kv_state_shape,
                        dtype=self._model.fprop_dtype,
                    ),
                }
            }
        }
        if not consolidate_rope_key_state:
          layer_kv_cache.update({
              'x_layers_{}'.format(i): {
                  'self_attention': {
                      'key_post_rotary_pos_emb': jnp.zeros(
                          kv_state_shape,
                          dtype=self._model.fprop_dtype,
                      ),
                  }
              }
          })
        kv_cache.update(layer_kv_cache)
      return {
          base_layer.DECODE_CACHE: {
              'lm': {
                  'time_step': self.input_sequence_len,
                  'transformer': kv_cache,
              }
          }
      }

    # decode cache sharding with a_blnh
    atten_ap = tr_atten_tpl.activation_split_dims_mapping
    if self.num_kv_heads == 1:
      if hasattr(atten_ap, 'blh'):
        kv_state_sharding = atten_ap.blh
      else:
        kv_state_sharding = (None, None, None)
    else:
      if hasattr(atten_ap, 'blnh'):
        kv_state_sharding = atten_ap.blnh
      else:
        kv_state_sharding = (None, None, None, None)
    kv_state_spec = base_layer.to_partition_spec(
        kv_state_sharding, self._model.mesh_axis_names
    )
    transformer_decode_partition_spec = {}
    for i in range(num_layers):
      layer_kv_spec = {
          'x_layers_{}'.format(i): {
              'self_attention': {
                  'key_state': kv_state_spec,
                  'value_state': kv_state_spec,
              }
          }
      }
      if not consolidate_rope_key_state:
        layer_kv_spec.update({
            'x_layers_{}'.format(i): {
                'self_attention': {'key_post_rotary_pos_emb': kv_state_spec}
            }
        })
      transformer_decode_partition_spec.update(layer_kv_spec)

    time_step_partition_spec = jax.sharding.PartitionSpec()

    decode_cache_pspecs = {
        base_layer.DECODE_CACHE: {
            'lm': {
                'time_step': time_step_partition_spec,
                'transformer': transformer_decode_partition_spec,
            }
        }
    }

    init_decode_cache_fn = jax_exp.pjit.pjit(
        _init_decode_cache_fn, out_shardings=decode_cache_pspecs
    )

    # update mdl_vars and mdl_var_pspecs
    self.decode_cache = init_decode_cache_fn()
    self.model_state.mdl_vars.update(self.decode_cache)

    self.decode_cache_pspecs = decode_cache_pspecs
    self.model_state.mdl_var_pspecs.update(self.decode_cache_pspecs)

  def init_decode_state(self):
    num_cache_slots = self.num_cache_slots

    decode_state = NestedMap()
    decode_state.start_step = jnp.array(
        [self.input_sequence_len], dtype=jnp.int32
    )
    decode_state.step = decode_state.start_step
    decode_state.per_sample_steps = jnp.ones(
        shape=num_cache_slots, dtype=jnp.int32
    ) * (self.input_sequence_len)
    decode_state.temperature = jnp.zeros(
        shape=num_cache_slots, dtype=jnp.float32
    )
    decode_state.per_example_max_decode_steps = (
        jnp.ones(shape=num_cache_slots, dtype=jnp.int32) * self.max_decode_steps
    )
    decode_state.per_example_top_p = jnp.ones(
        shape=num_cache_slots, dtype=jnp.float32
    )
    decode_state.per_example_top_k = jnp.ones(
        shape=num_cache_slots, dtype=jnp.int32
    )

    decode_state.done = jnp.ones(shape=num_cache_slots, dtype=jnp.bool_)
    decode_state.has_eos = jnp.zeros(shape=num_cache_slots, dtype=jnp.bool_)

    decode_state.decode_lengths = jnp.zeros(
        shape=num_cache_slots, dtype=jnp.int32
    )
    decode_state.prefix_lengths = jnp.zeros(
        shape=num_cache_slots, dtype=jnp.int32
    )
    decode_state.segment_pos = jnp.zeros(shape=num_cache_slots, dtype=jnp.int32)

    decode_state.output_ids = jnp.zeros(shape=num_cache_slots, dtype=jnp.int32)
    decode_state.logprobs = jnp.zeros(shape=num_cache_slots, dtype=jnp.float32)

    self.decode_state = decode_state

  def _register_bs_infos_for_input_shape(self, input_shape):
    batched_host_dummy = self.get_dummy_inputs(input_shape)
    batched_host_dummy = self.update_extra_inputs(
        batched_host_dummy,
        input_shape.batch_size,
        [self.default_extra_inputs] * input_shape.batch_size,
    )

    def _assert_type(x):
      assert isinstance(x, np.ndarray) or isinstance(
          x, jnp.ndarray
      ), f'Output of pre_processing contained an invalid type: {type(x)}'
      return x

    dummy_step = np.array(0, dtype=np.int32)
    dummy_prng_key = jax.random.PRNGKey(0)
    host_dummy = (
        dummy_step,
        dummy_prng_key,
        batched_host_dummy,
        self.get_nonbatch_inputs(batched_host_dummy),
    )
    host_dummy = jax.tree_util.tree_map(_assert_type, host_dummy)

    def _get_pspec(x):
      # Add a `cores` dimension.
      return jax.sharding.PartitionSpec(
          self.model_state.global_mesh.axis_names, *(None,) * (len(x.shape))
      )

    input_pspecs = jax.tree_util.tree_map(_get_pspec, host_dummy)
    num_cores = len(self.model_state.global_mesh.devices.flat)
    batched_input_psepcs = jax.tree_util.tree_map(
        _get_pspec, batched_host_dummy
    )

    global_inputs_shape_dtype = jax.tree_util.tree_map(
        lambda x: ((num_cores,) + x.shape, x.dtype), host_dummy
    )

    self._per_bs_infos[input_shape] = servable_model.MethodInputInfo(
        input_pspecs=input_pspecs,
        global_inputs_shape_dtype=global_inputs_shape_dtype,
    )
    info = self._per_bs_infos[input_shape]
    info.batched_input_pspecs = batched_input_psepcs

  def _register_for_input_shape(
      self, input_shape: servable_model.InputShapeInfo
  ):
    with self.model_state.global_mesh:
      # initialize kv cache and decode_state
      self.init_decode_cache()
      self.init_decode_state()
    # dummy tokens for generate_fn
    tokens = jnp.zeros((self.num_cache_slots,), dtype=jnp.int32)
    self._dummy_tokens_for_generate = (
        self.input_to_device_for_continuous_batching(
            tokens, InputShapeInfo(batch_size=self.num_cache_slots)
        )
    )
    # prefill device function
    prefill_input_shape = InputShapeInfo(1)
    self._register_bs_infos_for_input_shape(prefill_input_shape)
    self._prefill_device_fn = self._pjit_device_fn_prefill(
        self._per_bs_infos[prefill_input_shape].batched_input_pspecs
    )
    self._dummy_input_for_prefill = self.get_dummy_inputs(prefill_input_shape)
    self._dummy_input_for_prefill = self.update_extra_inputs(
        self._dummy_input_for_prefill,
        prefill_input_shape.batch_size,
        [self.default_extra_inputs] * prefill_input_shape.batch_size,
    )
    self._dummy_input_for_prefill = (
        self.input_to_device_for_continuous_batching(
            self._dummy_input_for_prefill, prefill_input_shape
        )
    )

    # insert device function
    self._insert_device_fn = self._pjit_device_fn_insert()

    # generate device function
    generate_input_shape = InputShapeInfo(input_shape.batch_size)
    self._register_bs_infos_for_input_shape(generate_input_shape)
    self._generate_device_fn = self._pjit_device_fn_generate()

    # warmup
    if self.model_state.precompile:
      logging.info('start precompile')
      slot_in_use = 0
      _, _, prefix_state = self.prefill_with_dummy()
      self.insert(prefix_state, slot_in_use)
      for _ in range(2):
        self.generate()  # compile w/ left_align_decode_state = False
        self.decode_state.step = jnp.array(
            [self.max_decode_steps + self.input_sequence_len - 1],
            dtype=jnp.int32,
        )
        self.generate()  # compile w/ left_align_decode_state = True
      self.prefill_with_dummy()
      self.insert(prefix_state, slot_in_use)
      # logging.info('start precompile')
      # _, _, prefix_state = self.prefill_with_dummy()
      # slot_in_use = 0
      # self.insert(prefix_state, slot_in_use)
      # self.generate()  # compile w/ left_align_decode_state = False
      # self.decode_state.step = jnp.array(
      #     [self.max_decode_steps + self.input_sequence_len - 1], dtype=jnp.int32
      # )
      # self.generate()  # compile w/ left_align_decode_state = True
      # # reset slot 0.
      # self.insert(prefix_state, slot_in_use)

  # JIT compiled generate function
  def _pjit_device_fn_generate(self):
    def _wrapped_fn_sample_generate(mdl_vars, decode_state, align_decode_state):
      mdl_vars = jax.tree_util.tree_map(
          jax.lax.with_sharding_constraint,
          mdl_vars,
          self.model_state.mdl_var_pspecs,
      )

      context_p = base_layer.JaxContext.HParams(do_eval=True)
      k1, k2 = jax.random.split(self._prng_key)
      with base_layer.JaxContext.new_context(hparams=context_p):

        def _model_fn(decode_state, align_decode_state):
          outputs = self.call_model_function_generate(
              decode_state, align_decode_state, mdl_vars, [k1, k2]
          )  # pytype: disable=wrong-arg-types  # jax-ndarray
          return outputs

        decode_state, decode_cache = _model_fn(decode_state, align_decode_state)
        return decode_state, decode_cache

    return jax_exp.pjit.pjit(
        _wrapped_fn_sample_generate,
        in_shardings=(self.model_state.mdl_var_pspecs, None),
        out_shardings=(None, self.decode_cache_pspecs),
        static_argnums=2,
        donate_argnums=(0,),
    )

  def call_model_function_generate(
      self, decode_state, align_decode_state, mdl_vars, prng_key
  ):
    k1, k2 = prng_key

    kwargs = {}
    if (
        'callback_device_index'
        in inspect.signature(self._model.decode_with_params).parameters
    ):
      kwargs['callback_device_index'] = self.callback_device_index

    decode_state, decode_cache = self._model.apply(
        mdl_vars,
        decode_state=decode_state,
        align_decode_state=align_decode_state,
        method=self._model.sample_generate,
        mutable=[
            base_layer.NON_TRAINABLE,
            base_layer.DECODE_CACHE,
            base_layer.PREFIX_DECODE_CACHE,
        ],
        rngs={
            base_layer.PARAMS: k1,
            base_layer.RANDOM: k2,
        },
        decoder_params=self._method_hparams.decoder,
        **kwargs,
    )
    return decode_state, decode_cache

  # JIT compiled insert function
  def _pjit_device_fn_insert(self):
    def _wrapped_fn_sample_insert(
        mdl_vars,
        prefix_decode_state,
        prefix_decode_cache,
        decode_state,
        slot,
    ):
      mdl_vars = jax.tree_util.tree_map(
          jax.lax.with_sharding_constraint,
          mdl_vars,
          self.model_state.mdl_var_pspecs,
      )

      context_p = base_layer.JaxContext.HParams(do_eval=True)
      k1, k2 = jax.random.split(self._prng_key)
      with base_layer.JaxContext.new_context(hparams=context_p):

        def _model_fn(
            prefix_decode_state,
            prefix_decode_cache,
            decode_state,
            slot,
        ):
          outputs = self.call_model_function_insert(
              prefix_decode_state,
              prefix_decode_cache,
              decode_state,
              slot,
              mdl_vars,
              [k1, k2],
          )  # pytype: disable=wrong-arg-types  # jax-ndarray

          return outputs

        decode_state, decode_cache = _model_fn(
            prefix_decode_state,
            prefix_decode_cache,
            decode_state,
            slot,
        )
        return decode_state, decode_cache

    return jax_exp.pjit.pjit(
        _wrapped_fn_sample_insert,
        in_shardings=(
            self.model_state.mdl_var_pspecs,
            None,
            self.decode_cache_pspecs,
            None,
            None,
        ),
        out_shardings=(None, self.decode_cache_pspecs),
        donate_argnums=(0,),
    )

  def call_model_function_insert(
      self,
      prefix_decode_state,
      prefix_decode_cache,
      decode_state,
      slot,
      mdl_vars,
      prng_key,
  ):
    k1, k2 = prng_key

    kwargs = {}
    if (
        'callback_device_index'
        in inspect.signature(self._model.decode_with_params).parameters
    ):
      kwargs['callback_device_index'] = self.callback_device_index

    decode_state, decode_cache = self._model.apply(
        mdl_vars,
        prefix_decode_state=prefix_decode_state,
        prefix_decode_cache=prefix_decode_cache,
        decode_state=decode_state,
        slot=slot,
        method=self._model.sample_insert,
        mutable=[
            base_layer.NON_TRAINABLE,
            base_layer.DECODE_CACHE,
            base_layer.PREFIX_DECODE_CACHE,
        ],
        rngs={
            base_layer.PARAMS: k1,
            base_layer.RANDOM: k2,
        },
        decoder_params=self._method_hparams.decoder,
        **kwargs,
    )
    return decode_state, decode_cache

  # JIT compiled prefill function
  def _pjit_device_fn_prefill(self, batched_input_pspecs):
    def _wrapped_fn_sample_prefill(mdl_vars, batched_inputs):
      mdl_vars = jax.tree_util.tree_map(
          jax.lax.with_sharding_constraint,
          mdl_vars,
          self.model_state.mdl_var_pspecs,
      )

      # Only one core has real data, others have zeros. Summing on the
      # leading `cores` dimension can make data replicated.
      def _replicate(x):
        return jax.lax.with_sharding_constraint(
            jnp.sum(x, axis=0, promote_integers=False), None
        )

      batched_inputs = jax.tree_util.tree_map(_replicate, batched_inputs)

      if self._model.fprop_dtype == jnp.bfloat16:
        # Convert float inputs/vars if fprop dtype is bfloat16.
        batched_inputs, mdl_vars = jax.tree_map(
            (lambda x: x.astype(jnp.bfloat16) if x.dtype == jnp.float32 else x),
            (batched_inputs, mdl_vars),
        )

      context_p = base_layer.JaxContext.HParams(do_eval=True)
      k1, k2 = jax.random.split(self._prng_key)
      with base_layer.JaxContext.new_context(hparams=context_p):

        def _model_fn(inputs):
          outputs = self.call_model_function_prefill(inputs, mdl_vars, [k1, k2])  # pytype: disable=wrong-arg-types  # jax-ndarray
          return outputs

        decode_state, decode_cache = _model_fn(batched_inputs)
        return decode_state, decode_cache

    # pjit-ed function.
    return jax_exp.pjit.pjit(
        _wrapped_fn_sample_prefill,
        in_shardings=(self.model_state.mdl_var_pspecs, batched_input_pspecs),
        out_shardings=(None, self.decode_cache_pspecs),
        donate_argnums=(0,),
    )

  def call_model_function_prefill(self, inputs, mdl_vars, prng_key):
    k1, k2 = prng_key

    kwargs = {}
    if (
        'callback_device_index'
        in inspect.signature(self._model.decode_with_params).parameters
    ):
      kwargs['callback_device_index'] = self.callback_device_index

    decode_state, decode_cache = self._model.apply(
        mdl_vars,
        input_batch=inputs,
        method=self._model.sample_prefill,
        mutable=[
            base_layer.NON_TRAINABLE,
            base_layer.DECODE_CACHE,
            base_layer.PREFIX_DECODE_CACHE,
        ],
        rngs={
            base_layer.PARAMS: k1,
            base_layer.RANDOM: k2,
        },
        decoder_params=self._method_hparams.decoder,
        **kwargs,
    )
    return decode_state, decode_cache

  def prefill(
      self, inputs: DeviceTensors
  ) -> tuple[DeviceTensors, DeviceTensors, DeviceTensors]:
    """Prefills the KV cache with the input sequence.

    Args:
      inputs: A single sequence of tokens (`prompt`) to run prefill on.

    Returns:
      token: Next token of the prompt, sampled by model's sampler.
      cache: Prefilled KV state.
    """
    # call device_fn prefill
    with self.model_state.global_mesh:
      prefix_decode_state, prefix_decode_cache = self._prefill_device_fn(
          self.model_state.mdl_vars, inputs
      )
      tokens = prefix_decode_state.output_ids
      scores = prefix_decode_state.logprobs
      return scores, tokens, (prefix_decode_state, prefix_decode_cache)

  def prefill_with_dummy(
      self,
  ) -> tuple[DeviceTensors, DeviceTensors, DeviceTensors]:
    """Prefills the KV cache with a dummy sequence. Used by secondary hosts.

    Returns:
      token: Next token of the prompt, sampled by model's default sampler.
      cache: Prefilled KV cache.
    """
    return self.prefill(self._dummy_input_for_prefill)

  def insert(self, prefix_state: DeviceTensors, slot: int) -> None:
    """Insert the prefix state into the specified slot of the target state.

    The target state is an internal state managed by the ServableMethod object.

    Args:
      prefix_state: the prefix kv state generated by prefill.
      slot: index of the cache slot to insert into.
    """
    # call device_fn insert to insert the prefill state to kv cache
    prefix_decode_state, prefix_decode_cache = prefix_state
    self.model_state.mdl_vars.update(self.decode_cache)
    logging.info('insert into slot %d', slot)
    with self.model_state.global_mesh:
      new_decode_state, new_decode_cache = self._insert_device_fn(
          self.model_state.mdl_vars,
          prefix_decode_state,
          prefix_decode_cache,
          self.decode_state,
          slot,
      )
      self.decode_state = new_decode_state
      self.decode_cache = new_decode_cache

  def generate(self) -> tuple[DeviceTensors, DeviceTensors, DeviceTensors]:
    """Given a batch of tokens and the KV state (managed internally), generate the next batch of tokens.

    Returns:
      new_tokens: a batch of new tokens sampled by model's sampler.
      done: a batch of booleans indicating whether the sampled token is EOS.
    """
    # update KV cache
    self.model_state.mdl_vars.update(self.decode_cache)
    left_align_decode_state = False
    if self.decode_state.step.addressable_data(0) == (
        self.max_decode_steps + self.input_sequence_len - 1
    ):
      logging.info('set left_align_decode_state to True')
      left_align_decode_state = True

    with self.model_state.global_mesh:
      decode_state, decode_cache = self._generate_device_fn(
          self.model_state.mdl_vars, self.decode_state, left_align_decode_state
      )
      new_tokens = decode_state.output_ids
      scores = decode_state.logprobs

      self.decode_state = decode_state
      self.decode_cache = decode_cache
      return scores, new_tokens, self.decode_state.done

  def detokenize(self, tokens: HostTensors) -> List[str]:
    """Detokenize a batch of sequences into a list of strings."""
    tokenizer = self._tokenizer

    # Use ragged tensor to get rid of the paddings
    # TODO(jwyang): fix potential bug that tokenizer don't ignore paddings
    def decode(ids_and_lens):
      ids, lens = ids_and_lens
      return tokenizer.IdsToStrings(tf.RaggedTensor.from_tensor(ids, lens))

    assert len(tokens.shape) == 2
    decode_lengths = np.sum(
        tokens != getattr(self._tokenizer.Vocabulary, 'pad_id', 0), axis=1
    )
    bytes_strs = np.array(decode((tokens, decode_lengths)))
    if isinstance(bytes_strs, bytes):
      bytes_strs = np.array([bytes_strs])
    return np.char.decode(bytes_strs.astype(np.bytes_), 'UTF-8')

  def resize_host_array(
      self,
      x: np.ndarray,
      global_input_shape_dtype: ShapesAndDtypes,
      unpadded_input_shape: InputShapeInfo,
  ):
    return jax_servable_model.ServableMethod.resize_host_array(
        self, x, global_input_shape_dtype, unpadded_input_shape
    )

  def input_to_device_for_continuous_batching(
      self, one_core_inputs: HostTensors, unpadded_shape: InputShapeInfo
  ) -> DeviceTensors:
    """Transfers input data to device."""

    # return self.input_to_device(one_core_inputs, unpadded_shape)
    def to_buffers(x):
      if self.model_state.is_primary_host:
        # Add a leading dimension for all-reduce across devices.
        x = x[np.newaxis, ...]
        zeros = np.zeros_like(x)
        # Only the first core on the primary host has the actual input.
        # The other cores on the primary host have zero inputs.
        return [
            jax.device_put(x if i == 0 else zeros, d)
            for i, d in enumerate(self._local_devices)
        ]
      else:
        # Add a leading dimension for all-reduce across devices.
        zeros = np.zeros((1,) + x.shape, x.dtype)
        # All cores on secondary hosts have zero inputs.
        return [jax.device_put(zeros, d) for d in self._local_devices]

    def jax_arrays_from_buffers(pspec, buffers, shape_dtype):
      shape, _ = shape_dtype
      return jax.make_array_from_single_device_arrays(
          shape,
          jax.sharding.NamedSharding(self.model_state.global_mesh, pspec),
          buffers,
      )

    pspecs = jax.tree_util.tree_map(
        lambda x: jax.sharding.PartitionSpec(
            self.model_state.global_mesh.axis_names, *(None,) * (len(x.shape))
        ),
        one_core_inputs,
    )
    num_cores = len(self.model_state.global_mesh.devices.flat)
    global_inputs_shape_dtype = jax.tree_util.tree_map(
        lambda x: ((num_cores,) + x.shape, x.dtype), one_core_inputs
    )

    host_inputs = jax.tree_util.tree_map(
        functools.partial(
            self.resize_host_array, unpadded_input_shape=unpadded_shape
        ),
        one_core_inputs,
        global_inputs_shape_dtype,
    )
    dummy_inputs_per_device_buffer = jax.tree_util.tree_map(
        to_buffers, host_inputs
    )
    dummy_inputs_on_host = jax.tree_util.tree_map(
        jax_arrays_from_buffers,
        pspecs,
        dummy_inputs_per_device_buffer,
        global_inputs_shape_dtype,
    )
    return dummy_inputs_on_host


class TextToEmbedding(servable_model.ServableMethod):
  """Implements text embedding method."""

  def __init__(
      self,
      model: base_model.BaseModel,
      model_fn_name: str,
      model_state: servable_model.ServableModelState,
      text_to_embedding_hparams: TextToEmbeddingHParams,
      tokenizer_p: Any,
      prng_key: PRNGKey,
      enable_auto_sharding: bool = False,
      compiler_options: jax.stages.CompilerOptions | None = None,
  ):
    self._tokenizer = tokenizer_p.Instantiate()
    self._text_to_embedding_hparams = text_to_embedding_hparams
    dummy_input_sample = ''
    # TODO(b/289379065): Remove this workaround here.
    if (
        isinstance(self._tokenizer, lm_tokenizer.LMTokenizer)
        and self._tokenizer.tokenized_input
    ):
      dummy_input_sample = '1'
    self._tf_sess_pre_processing = np_tf_sess_wrapper.wrap_tf_session(
        self.tf_pre_processing
    )
    super().__init__(
        model,
        model_fn_name,
        model_state,
        text_to_embedding_hparams,
        prng_key,
        dummy_input_sample,
        enable_auto_sharding=enable_auto_sharding,
        compiler_options=compiler_options,
    )

  @classmethod
  def service_id(cls) -> str:
    return lm_service.SERVICE_ID

  def fetch_output(
      self, model_fn_outputs: NestedJTensor, model_fn_inputs: NestedJTensor
  ) -> NestedJTensor:
    """Fetches useful output tensors from the model function outputs."""
    if not self._text_to_embedding_hparams.output_padding_name:
      return py_utils.NestedMap(
          text_embedding=model_fn_outputs[0][
              self._text_to_embedding_hparams.output_embedding_name
          ],
      )
    else:
      return py_utils.NestedMap(
          text_embedding=model_fn_outputs[0][
              self._text_to_embedding_hparams.output_embedding_name
          ],
          padding=model_fn_outputs[0][
              self._text_to_embedding_hparams.output_padding_name
          ],
      )

  def pre_processing(self, raw_inputs: List[str]) -> NestedNpTensor:
    """Preprocesses an unpadded batch of data into host numpy arrays."""
    prefixes = np.array(raw_inputs)
    # Provide an empty suffix per prefix so we can use the common tokenizer and
    # get the EOS token appended appropriately.
    suffixes = np.array(['' for _ in range(len(raw_inputs))])

    # HuggingFace tokenizer based custom vocabularies are enabled by applying
    # tf.py_function. The preprocessing and postprocessing are wrapped by
    # np_tf_sess_wrapper.wrap_tf_session function to allow export SavedModel.
    # However, the np_tf_sess_wrapper.wrap_tf_session function does not know how
    # to handle tf.py_function when trying to create a SavedModel-exportable
    # GraphDef object. Thus, to use custom vocabularies, we skip applying
    # np_tf_sess_wrapper.wrap_tf_session to preprocessing and postprocessing.
    if (
        isinstance(self._tokenizer, lm_tokenizer.LMTokenizer)
        and self._tokenizer.vocabulary_class
        and not self._tokenizer.tokenized_input
    ):
      tf_pre_processed = self.tf_pre_processing(prefixes, suffixes)
      return jax.tree_util.tree_map(np.array, tf_pre_processed)
    return self._tf_sess_pre_processing(prefixes, suffixes)

  def tf_pre_processing(
      self,
      prefixes: NestedNpOrTfTensor,
      suffixes: NestedNpOrTfTensor,
  ) -> NestedTfTensor:
    """Tokenizes `prefixes` and `suffixes` using TF ops.

    Args:
      prefixes: the prefix text batch of shape [batch_size].
      suffixes: the suffix text batch of shape [batch_size].

    Returns:
      A NestedMap of preprocessed tensors.
    """
    result = servable_lm_common.tf_tokenize_inputs(
        prefixes,
        suffixes,
        self._tokenizer,
        self._text_to_embedding_hparams.max_input_seq_len,
        self._text_to_embedding_hparams.max_suffix_seq_len,
        self._text_to_embedding_hparams.include_eos_score,
    )

    preprocessed = py_utils.NestedMap(
        ids=result.ids,
        labels=result.labels,
        paddings=result.paddings,
        weights=result.weights,
        inputs_indicator=result.inputs_indicator,
    )

    return preprocessed

  def post_processing(self, compute_outputs: NestedNpTensor) -> List[Any]:
    """Postprocesses the output numpy arrays to final host output."""
    if self._text_to_embedding_hparams.output_padding_name:
      paddings = compute_outputs['padding']  # [batch==1, max_seq_len]
      assert paddings.shape[0] == 1  # only supports batch_size == 1
      emb = compute_outputs['text_embedding']  # [batch==1, max_seq_len, dim]
      lengths = np.sum(1 - paddings, dtype=jnp.int32)  # Assume 1 is for pad
      emb_no_pad = emb[0, :lengths, :]  # [actual_seq_len, dim]
      return [emb_no_pad]
    else:
      return list(compute_outputs['text_embedding'])


class LMGradientMethod(ServableLMMethod):
  """Implements the gradient method of LM."""

  def __init__(
      self,
      model: base_model.BaseModel,
      model_state: servable_model.ServableModelState,
      prng_key: PRNGKey,
      gradient_params: GradientHParams,
      tokenizer_p: Any,
      exportable: bool = False,
      enable_auto_sharding: bool = False,
      compiler_options: jax.stages.CompilerOptions | None = None,
  ):
    self._tokenizer = tokenizer_p.Instantiate()
    self._gradient_params = gradient_params
    # gradient param contains all score params as well.
    # used for computing score
    self._score_params = gradient_params
    self._delimiter = '/'
    dummy_input_sample = ('', '')
    # TODO(b/289379065): Remove this workaround here.
    if (
        isinstance(self._tokenizer, lm_tokenizer.LMTokenizer)
        and self._tokenizer.tokenized_input
    ):
      dummy_input_sample = ('1', '1')
    logging.info(
        'Using np_tf_sess_wrapper on LMGradientMethod.tf_pre_processing'
    )
    self._tf_sess_pre_processing = np_tf_sess_wrapper.wrap_tf_session(
        # `bucketize_inputs` is only used in SavedModel export. The sax-native
        # serving has an equivalent bucketization after `pre_processing`.
        lambda *args: self.tf_pre_processing(*args, bucketize_inputs=False)
    )
    super().__init__(
        model,
        '__call__',
        model_state,
        gradient_params,
        prng_key,
        dummy_input_sample,
        exportable=exportable,
        enable_auto_sharding=enable_auto_sharding,
        compiler_options=compiler_options,
    )

  def call_model_function(
      self, inputs: NestedJTensor, mdl_vars: NestedJTensor, prng_key: PRNGKey
  ) -> NestedJTensor:
    tensors_to_take_gradients = {
        'inputs': {},
        'mdl_vars': {},
    }
    inputs_tensor_names = (
        self._gradient_params.inputs_tensor_names
        if self._gradient_params.inputs_tensor_names is not None
        else {}
    )
    mdl_vars_tensor_names = (
        self._gradient_params.mdl_vars_tensor_names
        if self._gradient_params.mdl_vars_tensor_names is not None
        else {}
    )
    split_inputs_tensor_names = {
        name: name.split(self._delimiter) for name in inputs_tensor_names
    }
    split_mdl_vars_tensor_names = {
        name: name.split(self._delimiter) for name in mdl_vars_tensor_names
    }

    def fetch(tree, keys):
      for key in keys:
        tree = tree[key]
      return tree

    def insert(tree, keys, x):
      for key in keys[:-1]:
        tree = tree[key]
      tree[keys[-1]] = x
      return x

    # preprocessed weights are incorrect for total loss computation in as it
    # sets all non-padding positions to 1. Score mask correctly sets only the
    # target positions to 1. Substitute weights with score mask for gradient.
    assert isinstance(inputs, dict)
    inputs['weights'] = inputs['score_masks']

    for k, v in split_inputs_tensor_names.items():
      try:
        # take only the first sample in the batch.
        tensors_to_take_gradients['inputs'][k] = fetch(inputs, v)[:1]
      except Exception as e:
        raise ValueError(f'Failed to find tensor {k} from inputs') from e
    for k, v in split_mdl_vars_tensor_names.items():
      try:
        tensors_to_take_gradients['mdl_vars'][k] = fetch(mdl_vars, v)
      except Exception as e:
        raise ValueError(f'Failed to find tensor {k} from mdl_vars') from e

    call_fn = super().call_model_function

    def forward_fn(tensors_to_take_gradients, inputs_no_grad, mdl_vars_no_grad):
      for k, v in tensors_to_take_gradients['inputs'].items():
        insert(inputs_no_grad, split_inputs_tensor_names[k], v)
      for k, v in tensors_to_take_gradients['mdl_vars'].items():
        insert(mdl_vars_no_grad, split_mdl_vars_tensor_names[k], v)
      outputs = call_fn(inputs_no_grad, mdl_vars_no_grad, prng_key)
      return outputs[0][0]['total_loss'][0], outputs

    compute_gradient_fn = jax.value_and_grad(
        forward_fn, has_aux=True, allow_int=True
    )
    (_, outputs), grads = compute_gradient_fn(
        tensors_to_take_gradients, inputs, mdl_vars
    )
    outputs = (outputs[0], outputs[1])  # 1 is for mutable.
    for key, value in grads['inputs'].items():
      if value.dtype is jax.dtypes.float0:
        # Gradient of an int-valued input cannot be consumed by jnp operation.
        # Zeros dtype should be int8 same as the original input that produced
        # float0.
        grads['inputs'][key] = jnp.zeros((), dtype=jnp.int8)
    outputs[0][0]['gradients'] = grads
    return outputs

  def fetch_output(
      self, model_fn_outputs: NestedJTensor, model_fn_inputs: NestedJTensor
  ) -> NestedJTensor:
    # fetch loss and gradients from the model output
    metrics, per_example_output = model_fn_outputs[0]
    batch_pad_size = model_fn_inputs['ids'].shape[0] - 1
    output = dict(
        # LMScore's fetch_output uses only the 0-th element of output.
        # per_example_output contains a 'scores' field by default from the loss
        # output, which will be the scores by default for fetch_output.
        # Models can retrieve intermediate per_token_xent from the forward pass
        # for fetch_output to mask out paddings.
        scores=LMScoreMethod.fetch_output(
            self, [per_example_output], model_fn_inputs
        ),
        # Pad total_loss with 0s to the shape (batch_size,)
        total_loss=jnp.array(
            [metrics['total_loss'][0]] + [0.0] * batch_pad_size
        ),
    )

    for grads_type, grads_dict in metrics['gradients'].items():
      for tensor_name, grads in grads_dict.items():
        output[f'gradients/{grads_type}/{tensor_name}'] = jnp.pad(
            # Provide a fake batch dim to mdl_vars to be consistent with inputs
            grads if grads_type == 'inputs' else grads.reshape(1, -1),
            pad_width=((0, batch_pad_size), (0, 0)),
        )

    return output

  def get_maxlen(self) -> int:
    return self._gradient_params.max_input_seq_len

  def output_seq_dim(self) -> int:
    return 1

  def pre_processing(self, raw_inputs: List[Tuple[str, str]]) -> NestedNpTensor:
    prefixes = np.array([prefix for prefix, _ in raw_inputs])
    suffixes = np.array([suffix for _, suffix in raw_inputs])

    # HuggingFace tokenizer based custom vocabularies are enabled by applying
    # tf.py_function. The preprocessing and postprocessing are wrapped by
    # np_tf_sess_wrapper.wrap_tf_session function to allow export SavedModel.
    # However, the np_tf_sess_wrapper.wrap_tf_session function does not know how
    # to handle tf.py_function when trying to create a SavedModel-exportable
    # GraphDef object. Thus, to use custom vocabularies, we skip applying
    # np_tf_sess_wrapper.wrap_tf_session to preprocessing and postprocessing.
    if (
        isinstance(self._tokenizer, lm_tokenizer.LMTokenizer)
        and self._tokenizer.vocabulary_class
        and not self._tokenizer.tokenized_input
    ):
      tf_pre_processed = self.tf_pre_processing(prefixes, suffixes)
      return jax.tree_util.tree_map(np.array, tf_pre_processed)
    return self._tf_sess_pre_processing(prefixes, suffixes)

  def post_processing(
      self, compute_outputs: NestedNpTensor
  ) -> List[Dict[str, List[float]]]:
    flattened_outputs = jax.tree_util.tree_map(
        lambda x: x.flatten().tolist(), compute_outputs
    )

    return [flattened_outputs]  # The extra list is to just conform to base api.

  def tf_pre_processing(
      self,
      prefixes: NestedNpOrTfTensor,
      suffixes: NestedNpOrTfTensor,
      extra_inputs: Mapping[str, Any] | None = None,
      branch_index: NestedNpOrTfTensor | None = None,
      bucketize_inputs: bool = True,
  ) -> NestedTfTensor:
    """Tokenizes `prefixes` and `suffixes` using TF ops.

    This also implements `ExportableToSavedModel.tf_pre_processing`.

    Args:
      prefixes: the prefix text batch of shape [batch_size].
      suffixes: the suffix text batch of shape [batch_size].
      extra_inputs: optional mapping of extra input key to tensor or tensor spec
        of shape [batch_size].
      branch_index: optional index to indicate which bucket key will be used by
        `bucketize_tokenized_inputs`.
      bucketize_inputs: whether to bucketize the preprocessed inputs based on
        max sequence length in the batch.

    Returns:
      A NestedMap of preprocessed tensors.
    """
    preprocessed = servable_lm_common.tf_tokenize_inputs(
        prefixes,
        suffixes,
        self._tokenizer,
        self._gradient_params.max_input_seq_len,
        self._gradient_params.max_suffix_seq_len,
        self._gradient_params.include_eos_score,
    )

    if bucketize_inputs:
      preprocessed = servable_lm_common.bucketize_tokenized_inputs(
          self.sorted_seq_lens,
          preprocessed,
          branch_index,
      )

    if extra_inputs:
      preprocessed.update(extra_inputs)

    return preprocessed

  def tf_post_processing(self, outputs: NestedTfTensor) -> NestedTfTensor:
    if self._gradient_params.mdl_vars_tensor_names:
      raise ValueError(
          'Exporting graident method with gradients to model '
          'variables is not supported since it is undefined '
          'how to introduce the batch dims for export signatures.'
      )

    return outputs

  def input_signature(
      self, batch_size: Optional[int]
  ) -> Tuple[TensorSpec, TensorSpec, Mapping[str, TensorSpec], TensorSpec]:
    """Implements `ExportableToSavedModel.input_signature`."""
    return (
        tf.TensorSpec([batch_size], dtype=tf.string, name='prefixes'),
        tf.TensorSpec([batch_size], dtype=tf.string, name='suffixes'),
        servable_lm_common.extra_inputs_to_tf_signature(
            self._extra_inputs,
            batch_size,
            self.method_params.extra_inputs_dtypes,
        ),
        oex.TensorSpecWithDefault(
            tf.TensorSpec(
                [batch_size], dtype=tf.int32, name='branch_index_warmup_only'
            ),
            tf.constant([-1], dtype=tf.int32, shape=[batch_size or 1]),
        ),
    )

  @property
  def tf_trackable_resources(self) -> Any:
    """Implements `ExportableToSavedModel.tf_trackable_resources`."""
    return None


class ServableLMModel(servable_model.ServableModel):
  """Represents an implementation for the LM service, backed by a model.

  This class is responsible for model loading, batch padding, etc.
  """

  def init_method(
      self,
      method: str,
      model: base_model.BaseModel,
      model_state: servable_model.ServableModelState,
      method_params: servable_model_params.ServableMethodParams,
      prng_key: PRNGKey,
  ) -> servable_model.ServableMethod:
    assert isinstance(self.model_config, ServableLMModelParams)
    tokenizer_p = self.model_config.serving_tokenizer()
    if method == LMMethodName.SCORE:
      assert isinstance(method_params, ScoreHParams)
      return LMScoreMethod(
          model,
          model_state,
          prng_key,
          method_params,
          tokenizer_p,
          exportable=True,
          enable_auto_sharding=self._enable_auto_sharding,
          compiler_options=self._compiler_options,
      )
    elif method == LMMethodName.GENERATE:
      assert isinstance(method_params, DecodeHParams)
      if method_params.decoder.num_cache_slots > 0:
        return LMDecodeMethodContinuousBatching(
            model,
            model_state,
            prng_key,
            method_params,
            tokenizer_p,
            exportable=True,
            enable_auto_sharding=self._enable_auto_sharding,
            compiler_options=self._compiler_options,
        )
      else:
        return LMDecodeMethod(
            model,
            model_state,
            prng_key,
            method_params,
            tokenizer_p,
            exportable=True,
            enable_auto_sharding=self._enable_auto_sharding,
            compiler_options=self._compiler_options,
        )
    elif method == LMMethodName.GENERATE_STREAM:
      assert isinstance(method_params, DecodeHParams)
      return LMDecodeMethod(
          model,
          model_state,
          prng_key,
          method_params,
          tokenizer_p,
          exportable=False,
          streamable_output=True,
          enable_auto_sharding=self._enable_auto_sharding,
          compiler_options=self._compiler_options,
      )
    elif method == LMMethodName.EMBED:
      assert isinstance(method_params, TextToEmbeddingHParams)
      assert method_params.output_embedding_name is not None
      if method_params.model_method_name is None:
        raise ValueError(
            'Must specify `model_method_name` in TextToEmbeddingHParams.'
        )
      return TextToEmbedding(
          model,
          method_params.model_method_name,
          model_state,
          method_params,
          tokenizer_p,
          prng_key=prng_key,
          enable_auto_sharding=self._enable_auto_sharding,
          compiler_options=self._compiler_options,
      )
    elif method == LMMethodName.GRADIENT:
      assert isinstance(method_params, GradientHParams)
      assert (
          method_params.inputs_tensor_names is not None
          or method_params.mdl_vars_tensor_names is not None
      )
      return LMGradientMethod(
          model,
          model_state,
          prng_key,
          method_params,
          tokenizer_p,
          exportable=True,
          enable_auto_sharding=self._enable_auto_sharding,
          compiler_options=self._compiler_options,
      )
    else:
      raise NotImplementedError(f'method {method} not implemented')

  def supports_dummy_compute_on_primary(self) -> bool:
    if self.methods is None or not isinstance(self.methods, Dict):
      return True
    for method in list(self.methods.values()):
      has_multiple_seq_lens = (
          hasattr(method, 'sorted_seq_lens')
          and method.sorted_seq_lens is not None
          and len(method.sorted_seq_lens) > 1
      )
      if has_multiple_seq_lens:
        return False
    return True

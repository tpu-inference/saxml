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
"""Customize layers for sax."""
import functools
from typing import Optional, Tuple

from flax import linen as nn
import jax
from jax import numpy as jnp
import numpy as np
from praxis import base_layer
from praxis import layers
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes
from praxis.layers import embedding_softmax
from praxis.layers import multi_query_attention


template_field = base_layer.template_field
JTensor = pytypes.JTensor
LayerTpl = pax_fiddle.Config[base_layer.BaseLayer]
NestedMap = py_utils.NestedMap
DECODE_CACHE = base_layer.DECODE_CACHE
CACHE_SCALE_SUFFIX = '_scale'


def reduce_last_dim_for_quantization(t: JTensor) -> tuple[JTensor, JTensor]:
  bound = jnp.max(jnp.abs(t), axis=[-1], keepdims=True)
  scale = bound / 127.0
  scale = jnp.where(scale == 0.0, 1.0, scale)
  t = jnp.round(jnp.divide(t, scale))
  t = jnp.clip(t, -127.0, 127.0).astype(jnp.int8)
  return t, scale


class LLaMARotaryEmbedding(embedding_softmax.RotaryPositionalEmbedding):
  """LLaMA variant of ROPE where inputs are split in a different way."""

  def __call__(
      self,  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
      inputs: JTensor,
      position: Optional[JTensor] = None,
  ) -> JTensor:
    """Generates a JTensor of sinusoids with different frequencies.

    Args:
      inputs: The input sequence on which to apply the Rotary position
        embedding. Since rotary position embeddings are applied to query and
        keys after projection, it is assumed of shape [B, S, N, H].
      position: Optional position JTensor which denotes the position of each
        token in the sequence. This only needs to be supplied when the sequence
        is packed. It is of shape [B, S].

    Returns:
      a JTensor of shape [B, S, N, H] which includes the inputs together with
      the rotary position embedding incorporated in it.
    """
    if len(inputs.shape) != 4:
      raise ValueError(
          'Input is assumed to be a rank 4 tensor of shape'
          '[batch, sequence, heads, dims].'
      )
    if self.embedding_dims != inputs.shape[3]:
      raise ValueError(
          'The embedding dims of the rotary position embedding'
          'must match the hidden dimension of the inputs.'
      )
    inputs_shifted_left = jnp.concatenate(
        [inputs[..., 1:], inputs[..., :1]], axis=-1
    )
    inputs_shifted_right = jnp.concatenate(
        [inputs[..., -1:], inputs[..., :-1]], axis=-1
    )
    inputs_shifted = jax.lax.select(
        jnp.tile(
            jnp.mod(jnp.arange(self.embedding_dims, dtype=jnp.int32), 2),
            inputs.shape[:-1] + (1,),
        ),  # [[[[0, 1, 0, 1, ...], ...]
        inputs_shifted_right,
        inputs_shifted_left,
    )
    half_embedding_dim = self.embedding_dims // 2
    fraction = 2 * jnp.arange(0, half_embedding_dim) / self.embedding_dims
    fraction = jnp.repeat(fraction, 2)
    timescale = (
        self.min_timescale
        * (self.max_timescale / self.min_timescale) ** fraction
    )
    if position is None:
      seq_length = inputs.shape[1]
      position = jnp.arange(seq_length, dtype=jnp.float32)[jnp.newaxis, :]
    position = position[:, :, jnp.newaxis, jnp.newaxis]
    timescale = timescale[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]
    sinusoid_inp = position / timescale
    sin = jnp.sin(sinusoid_inp)
    cos = jnp.cos(sinusoid_inp)
    sign = jnp.sign(
        jnp.mod(jnp.arange(self.embedding_dims, dtype=jnp.int32), 2) - 0.5
    )  # [-1, 1, -1, 1, ...]
    outputs = inputs * cos + inputs_shifted * sin * sign
    if self.cast_as_fprop_dtype:
      outputs = outputs.astype(self.fprop_dtype)
    return outputs


class FakeLayerNorm(layers.LayerNorm):

  def setup(self) -> None:
    return

  def __call__(self, inputs, paddings=None):
    return inputs


class TransformerMLP(layers.TransformerFeedForward):
  ln_tpl: LayerTpl = template_field(FakeLayerNorm)


# TODO(huangyp): adapt the more efficient lingvo implementation.
class ParallelTransformer(layers.Transformer):
  """Transformer with parallel attention and feedforward."""

  norm_policy = 'pre'  # Use primer_hybrid for GPT-Neo
  residual_droppath_prob = 0.0
  use_cross_attention = False
  tr_fflayer_tpl: LayerTpl = template_field(TransformerMLP)

  def ffn_norm(self, inputs: JTensor, inputs_normalized: JTensor) -> JTensor:
    # Apply FFN layer
    if self.norm_policy == 'primer_hybrid':
      ffn_inputs = self.post_layer_norm(inputs)
    elif self.norm_policy == 'pre':
      ffn_inputs = inputs_normalized
    else:
      ffn_inputs = inputs
    return ffn_inputs

  def __call__(
      self,
      inputs: JTensor,
      paddings: JTensor,
      attention_mask: JTensor,
      cross_inputs: Optional[JTensor] = None,
      cross_attention_mask: Optional[JTensor] = None,
      segment_pos: Optional[JTensor] = None,
      segment_ids: Optional[JTensor] = None,
  ) -> Tuple[JTensor, JTensor]:
    """Transformer decoder layer for GPT-J and NeoX.

    Args:
      inputs: Input sequence JTensor of shape [B, T, H].
      paddings: Input paddings JTensor of shape [B, T] (only used in FFN layer).
      attention_mask: Self attention mask ready to add to the logits. It can be
        of shape [1|B, 1, 1|T, T] which is broadcast compatible with the self
        attention matrix of shape [B, N, T, T]. This is assumed to have combined
        paddings, causal masking as well as segment maskings.
      cross_inputs: Output of the encoder, to be used for cross attention, of
        shape [B, S, H].
      cross_attention_mask: Cross attention mask ready to add to the logits. It
        can be of shape [1|B, 1, 1|T, S] which is broadcast compatible with the
        cross attention matrix of shape [B, N, T, S]. This is assumed to have
        combined paddings as well as segment maskings.
      segment_pos: A JTensor of shape [B, T]. The position of each token in a
        segment.
      segment_ids: A JTensor of shape [B, T] specifying which segment each token
        belongs to.

    Returns:
      The fflayer output with shape [B, T, D].
      atten_probs: A NestedMap with keys `self_atten` <float>[B, N, T, T].
    """
    assert not self.use_cross_attention
    assert self.residual_droppath_prob == 0.0
    if self.norm_policy == 'primer_hybrid':
      inputs_normalized = self.pre_layer_norm(inputs)
    elif self.norm_policy == 'pre':
      inputs_normalized = self.layer_norm(inputs)
    else:
      inputs_normalized = inputs
    # Compute self-attention, key/value vectors are the input itself
    atten_output, self_atten_probs = self.self_attention(
        inputs_normalized,
        inputs_normalized,
        inputs_normalized,
        atten_mask=attention_mask,
        query_segment_pos=segment_pos,
        key_segment_pos=segment_pos,
    )
    atten_probs = NestedMap(self_atten=self_atten_probs)

    # Residual dropout and connection
    atten_output = self.residual_dropout(atten_output)

    ffn_inputs = self.ffn_norm(inputs, inputs_normalized)
    ffn_output = self.ff_layer(ffn_inputs, paddings=paddings)
    output = atten_output + ffn_output + inputs
    return output, atten_probs  # pytype: disable=bad-return-type  # jax-ndarray

  def extend_step(
      self,
      inputs: JTensor,
      *,
      time_step: JTensor,
      segment_pos: Optional[JTensor] = None,
      attention_mask: JTensor,
      cross_attention_mask: Optional[JTensor] = None
  ) -> JTensor:
    # pyformat:disabled
    """Transformer decoder layer, autoregressive cached decoding.

    For cross attention, the key/value cache may have a smaller batch size b
    than inputs batch size B. In this case, we require B % b == 0, and this
    corresponds to multi-sample decoding for each input in b, and cross-
    attention states will be repeated by (B // b) times. Each consecutive
    (B // b) chunk in B correspond to multiple samples for the same cross
    # inputs.

    When `inputs` has shape [B, D], it will do extend_step on one token per
    batch in regular autoregressive decoding.

    When `inputs` has shape [B, L, D], it will do extend_step on L tokens per
    batch. This is used to do suffix scoring after autoregressive decoding.

    Args:
      inputs:         [B, D] or [B, L, D], target sequence at index time_step.
      time_step:      a 0-based scalar, the current decode step.
      segment_pos:    [B] or [B, L], the current position in the same segment.
        If unspecified, time_step will be used.
      attention_mask: [B, 1, L, S] if extends multiple steps (i.e. `inputs` is
        of shape [B, L, D]) or [B, 1, T] if extends one step (i.e. `inputs` is
        of shape [B, D]), optional attention mask for this time step. This
        combines causal mask with any segment mask if applicable.
      cross_attention_mask: [b|B, 1, 1 S], optional, cross_segment_mask for this
        time step. This combines padding mask with any segment mask if
        applicable.

    Returns:
      output: [B, D] or [B, L, D].
    """
    # Layer normalize input
    if self.norm_policy == 'primer_hybrid':
      inputs_normalized = self.pre_layer_norm(inputs)
    elif self.norm_policy == 'pre':
      inputs_normalized = self.layer_norm(inputs)
    else:
      inputs_normalized = inputs

    # Self-attention layer.
    atten_output = self.self_attention.extend_step(
        inputs_normalized,
        atten_mask=attention_mask,
        time_step=time_step,
        segment_pos=segment_pos,
    )

    # Residual dropout and connection
    atten_output = self.residual_dropout(atten_output)
    # Apply FFN layer
    ffn_inputs = self.ffn_norm(inputs, inputs_normalized)
    ffn_output = self.ff_layer.extend_step(ffn_inputs, time_step=time_step)
    output = atten_output + ffn_output + inputs
    return output


class ParallelTransformerOnlyNormAttentionInputs(ParallelTransformer):
  """Transformer with parallel attention and feedforward."""

  def ffn_norm(self,
               inputs: JTensor,
               inputs_normalized: JTensor) -> JTensor:
    # Apply FFN layer
    if self.norm_policy == 'primer_hybrid':
      ffn_inputs = self.post_layer_norm(inputs)
    else:
      ffn_inputs = inputs_normalized
    return ffn_inputs


class GPTJRotaryEmbedding(embedding_softmax.RotaryPositionalEmbedding):
  """GPTJ variant of ROPE where rotary_dim != dim_per_head."""

  max_position_embeddings: Optional[int] = None
  rotary_dim: Optional[int] = None

  def setup(self) -> None:
    super().setup()
    self.embed_positions = self.create_sinusoidal_positions(
        self.max_position_embeddings, self.rotary_dim
    )

  def create_sinusoidal_positions(self, num_pos, dim):
    inv_freq = 1.0 / (10000 ** (jnp.arange(0, dim, 2) / dim))
    sinusoid_inp = jnp.einsum(
        'i , j -> i j', jnp.arange(num_pos), inv_freq
    ).astype('float32')
    sin, cos = jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)
    return jnp.concatenate((sin, cos), axis=-1)

  def rotate_every_two(self, tensor):
    rotate_half_tensor = jnp.stack(
        (-tensor[:, :, :, 1::2], tensor[:, :, :, ::2]), axis=-1
    )
    rotate_half_tensor = rotate_half_tensor.reshape(
        rotate_half_tensor.shape[:-2] + (-1,)
    )
    return rotate_half_tensor

  def apply_rotary_pos_emb(self, tensor, sin_pos, cos_pos):
    sin_pos = sin_pos[:, :, None, :].repeat(2, 3)
    cos_pos = cos_pos[:, :, None, :].repeat(2, 3)
    return (tensor * cos_pos) + (self.rotate_every_two(tensor) * sin_pos)

  def __call__(
      self,  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
      inputs: JTensor,
      position: Optional[JTensor] = None,
  ) -> JTensor:
    """Generates a JTensor of sinusoids with different frequencies.

    Args:
      inputs: The input sequence on which to apply the Rotary position
        embedding. Since rotary position embeddings are applied to query and
        keys after projection, it is assumed of shape [B, S, N, H].
      position: Optional position JTensor which denotes the position of each
        token in the sequence. This only needs to be supplied when the sequence
        is packed. It is of shape [B, S].

    Returns:
      a JTensor of shape [B, S, N, H] which includes the inputs together with
      the rotary position embedding incorporated in it.
    """
    if len(inputs.shape) != 4:
      raise ValueError(
          'Input is assumed to be a rank 4 tensor of shape'
          '[batch, sequence, heads, dims].'
      )

    if position is None:
      seq_length = inputs.shape[1]
      position = jnp.arange(seq_length, dtype=jnp.int32)[jnp.newaxis, :]

    sincos = jnp.take(self.embed_positions, position, axis=0)
    sincos = jnp.split(sincos, 2, axis=-1)
    sin, cos = sincos
    inp_rot = inputs[:, :, :, : self.rotary_dim]
    inp_pass = inputs[:, :, :, self.rotary_dim :]
    inp_rot = self.apply_rotary_pos_emb(inp_rot, sin, cos)
    first_part = inp_rot
    second_part = inp_pass
    if self.cast_as_fprop_dtype:
      first_part = first_part.astype(self.fprop_dtype)
      second_part = second_part.astype(self.fprop_dtype)
    return jnp.concatenate([first_part, second_part], axis=-1)


class QuantizedKVMQA(multi_query_attention.MultiQueryDotProductAttention):
  """MQA with INT8 KV states."""

  quantize_kv: bool = False  # Whether to quantize kv cache.

  def update_decode_state(self, name: str, state: JTensor) -> None:
    if not self.quantize_kv:
      return super().update_decode_state(name, state)

    if not self.is_mutable_collection(DECODE_CACHE):
      return

    q_value, q_scale = reduce_last_dim_for_quantization(state)
    self.put_variable(DECODE_CACHE, name, q_value)
    self.put_variable(DECODE_CACHE, name + CACHE_SCALE_SUFFIX, q_scale)

  def get_decode_state(self, name: str) -> JTensor:
    if not self.quantize_kv:
      return super().get_decode_state(name)

    q_value = self.get_variable(DECODE_CACHE, name)
    q_scale = self.get_variable(DECODE_CACHE, name + CACHE_SCALE_SUFFIX)
    return jnp.multiply(q_value, q_scale)

  @nn.nowrap
  def extend_decode_state(
      self, name: str, value: JTensor, time_step: JTensor, time_dim: int
  ) -> JTensor:
    if not self.quantize_kv:
      return super().extend_decode_state(name, value, time_step, time_dim)

    if (self.num_kv_heads == 1 and len(value.shape) == time_dim + 1) or (
        self.num_kv_heads > 1 and len(value.shape) == time_dim + 2
    ):
      extend_value = jnp.expand_dims(value, axis=time_dim)
    else:
      extend_value = value
    qvalue, qscale = reduce_last_dim_for_quantization(extend_value)
    indices = [0] * qvalue.ndim
    indices[time_dim] = time_step.astype(jnp.int32)
    state = self.get_variable(DECODE_CACHE, name)
    assert state is not None
    scale = self.get_variable(DECODE_CACHE, name + CACHE_SCALE_SUFFIX)
    assert scale is not None

    new_state = jax.lax.dynamic_update_slice(
        state, qvalue.astype(state.dtype), indices
    )
    new_scale = jax.lax.dynamic_update_slice(
        scale, qscale.astype(scale.dtype), indices
    )
    if self.is_mutable_collection(DECODE_CACHE):
      self.put_variable(DECODE_CACHE, name, new_state)
      self.put_variable(DECODE_CACHE, name + CACHE_SCALE_SUFFIX, new_scale)
    return new_state

  def transform_decode_state(
      self, transform_fn: base_layer.DecodeStateTransformFn
  ):
    """Transforms all decode state variables based on transform_fn."""
    batch_dim = 0
    time_dim = 1
    names = self.variables[base_layer.DECODE_CACHE].keys()
    for name in names:
      if CACHE_SCALE_SUFFIX in name:
        continue
      state = self.get_decode_state(name)
      if not isinstance(state, JTensor):
        continue
      new_state = transform_fn(state, batch_dim, time_dim)
      if self.num_kv_heads == 1:
        new_state = self._shard_blh(new_state)
      else:
        new_state = self._shard_blnh(new_state)
      self.update_decode_state(name, new_state)


class TransformerFeedForwardWithSeqSplit(layers.TransformerFeedForward):
  """TransformerFeedForward with seq split."""

  chunked_ffn_num_seq_split: int = 16

  def _compute_ffns(self, inputs, paddings, ap_ff0=None):
    assert self._is_ffn1_gated
    if (
        not self.do_eval
        or inputs.ndim == 2
        or self.chunked_ffn_num_seq_split == 1
        or inputs.shape[1] < self.chunked_ffn_num_seq_split
    ):
      return super()._compute_ffns(inputs, paddings, ap_ff0)
    t = inputs.shape[1]  # time dimension
    assert (
        t % self.chunked_ffn_num_seq_split == 0
    ), f'{t=} must divide {self.chunked_ffn_num_seq_split}'
    chunk_size = t // self.chunked_ffn_num_seq_split

    val = NestedMap()
    val.inputs = inputs
    val.paddings = paddings
    # prefix_ids are right aligned while paddings are on the left
    val.i = self.chunked_ffn_num_seq_split - 1

    def cond_fn(mdl, val):
      del mdl
      # Check paddings at rigthmost position in the chunk are 0 or not.
      end = (val.i + 1) * chunk_size - 1
      return jnp.logical_and(
          val.i >= 0, jnp.sum(val.paddings[:, end, 0].astype(jnp.int32)) == 0
      )

    def loop_body(mdl, val):
      start = val.i * chunk_size
      chunk_inputs = jax.lax.dynamic_slice_in_dim(
          val.inputs, start, chunk_size, axis=1
      )
      chunk_padings = jax.lax.dynamic_slice_in_dim(
          val.paddings, start, chunk_size, axis=1
      )
      gate_value = mdl.ffn_layer1_gate(chunk_inputs)
      # theta.ffn_layer1 corresponds to gshard_builder's wi1
      activations = gate_value * mdl.ffn_layer1(chunk_inputs)

      # Apply paddings if not None
      if chunk_padings is not None:
        activations *= 1.0 - chunk_padings

      # Apply second FFN layer
      outputs = mdl.ffn_layer2(activations)
      val.inputs = jax.lax.dynamic_update_slice(
          val.inputs, outputs, [0, start, 0]
      )
      val.i -= 1
      return val

    val = nn.while_loop(cond_fn, loop_body, self, val)
    return val.inputs


class ChunkedMQA(QuantizedKVMQA):
  """MQA that computes qkv attention only on non-padded slices.

  A lot of time positions of kv states are just paddings. We can identify the
  start and slice widths of non-padding kv states by checking atten_mask and
  the present time_step during extend_step. We then  do qk_einsum and
  pv_enisum only over the non-padding slices. However, jax
  won't allow dynamic_slice with variable widths, we have to predefine
  `chunked_one_step_attn_num_seq_split` number of partial functions, each with
  fixed slice width, and use jax.lax.switch to pick the corresponding function
  based on the actual dynamic non-padding width from the input batch.
  """

  chunked_one_step_attn_num_seq_split: int = 16

  def _dot_atten_one_step_from_qkv(
      self,
      query: JTensor,
      key: JTensor,
      value: JTensor,
      atten_mask: JTensor,
      relative_bias: JTensor | None = None,
      time_step: JTensor | None = None,
  ) -> tuple[JTensor, JTensor]:
    if (
        self.num_kv_heads > 1
        or self.chunked_one_step_attn_num_seq_split <= 1
        or relative_bias is not None
        or len(query.shape) != 3
    ):
      return super()._dot_atten_one_step_from_qkv(
          query, key, value, atten_mask, relative_bias
      )
    b, s, h = key.shape
    base_layer.assert_has_shape(query, [b, -1, h])
    base_layer.assert_has_shape(atten_mask, [-1, -1, s])
    base_layer.assert_has_shape(value, [b, s, h])
    query = self._scale_query(query)

    assert s % self.chunked_one_step_attn_num_seq_split == 0
    w = s // self.chunked_one_step_attn_num_seq_split

    def _dynamic_qkv(query, key, value, atten_mask, start_chunk, num_chunks):
      key = jax.lax.dynamic_slice(
          key, [0, start_chunk, 0], [b, num_chunks * w, h]
      )
      value = jax.lax.dynamic_slice(
          value, [0, start_chunk, 0], [b, num_chunks * w, h]
      )
      atten_mask = jax.lax.dynamic_slice(
          atten_mask, [0, 0, start_chunk], [b, 1, num_chunks * w]
      )
      logits = self.qk_einsum('BNH,BSH->BNS', query, key)
      logits = self._cap_logits(logits)
      # Attention softmax is always carried out in fp32.
      logits = logits.astype(jnp.float32)
      # Apply attention masking
      padded_logits = py_utils.apply_mask_to_logits(logits, atten_mask)
      # Of shape [b, n, s]
      if self.attention_extra_logit is None:
        probs = jax.nn.softmax(padded_logits, axis=-1).astype(key.dtype)
      else:
        probs = jnp.exp(
            self._log_softmax_with_extra_logit(padded_logits)
        ).astype(key.dtype)

      return self.pv_einsum('BNS,BSH->BNH', probs, value)

    dynamic_qkv_fns = []
    for c in range(self.chunked_one_step_attn_num_seq_split):
      dynamic_qkv_fns.append(functools.partial(_dynamic_qkv, num_chunks=c + 1))

    start_chunk = 0
    num_chunks = time_step // w - start_chunk + 1
    encoded = jax.lax.switch(
        num_chunks - 1,
        dynamic_qkv_fns,
        query,
        key,
        value,
        atten_mask,
        start_chunk,
    )

    return encoded, jnp.zeros((0,))  # pytype: disable=bad-return-type  # jax-ndarray


class ChunkedMHA(layers.DotProductAttention):
  """MHA that computes qkv attention only on non-padded slices.

  A lot of time positions of kv states are just paddings. We can identify the
  start and slice widths of non-padding kv states by checking atten_mask and
  the present time_step during extend_step. We then  do qk_einsum and
  pv_enisum only over the non-padding slices. However, jax
  won't allow dynamic_slice with variable widths, we have to predefine
  `chunked_one_step_attn_num_seq_split` number of partial functions, each with
  fixed slice width, and use jax.lax.switch to pick the corresponding function
  based on the actual dynamic non-padding width from the input batch.
  """

  chunked_one_step_attn_num_seq_split: int = 16

  def _dot_atten_one_step(
      self,
      query: JTensor,
      key_state_name: str,
      value_state_name: str,
      atten_mask: JTensor,
      relative_bias: JTensor | None = None,
      time_step: JTensor | None = None,
  ) -> tuple[JTensor, JTensor]:
    """Dot attention function for queries with 1 time step.

    Args:
      query: JTensor of shape [B, N, H].
      key_state_name: Name of the decoding key state variable.
      value_state_name: Name of the decoding value state variable.
      atten_mask: JTensor of shape [1|B, 1, S] which is a mask that is applied
        to prevent attention between unwanted pairs. This has already been
        converted into large negative logits. The first dimension is allowed to
        be of size 1, if the mask is shared by all items in the batch (e.g.,
        only a causal mask).
      relative_bias: Relative bias of shape [1|B, N, 1, S].
      time_step: A scalar. The time step tensor.

    Returns:
      encoded: JTensor of shape [B, N, H].
      probs: JTensor of shape [B, N, S].
    """
    if (
        self.chunked_one_step_attn_num_seq_split <= 1
        or relative_bias is not None
    ):
      return super()._dot_atten_one_step(
          query,
          key_state_name,
          value_state_name,
          atten_mask,
          relative_bias,
          time_step,
      )
    key = self._shard_blnh(self.get_decode_state(key_state_name))
    value = self._shard_blnh(self.get_decode_state(value_state_name))
    k_b = key.shape[0]
    q_b = query.shape[0]
    if q_b != k_b:
      if q_b % k_b != 0:
        raise ValueError(
            f'q batch size {q_b} is not divisible by state batch size {k_b}'
        )
      key = jnp.repeat(key, q_b // k_b, axis=0)
      value = jnp.repeat(value, q_b // k_b, axis=0)
    if atten_mask.shape[0] != q_b and atten_mask.shape[0] != 1:
      assert atten_mask.shape[0] == k_b, (atten_mask.shape, k_b)
      atten_mask = jnp.repeat(atten_mask, q_b // k_b, axis=0)
    # query is 3d.
    query = self._shard_bnh(query)

    b, s, n, h = key.shape
    base_layer.assert_has_shape(value, [b, s, n, h])
    base_layer.assert_has_shape(query, [b, n, h])
    base_layer.assert_has_shape(atten_mask, [-1, 1, s])
    query = self._scale_query(query)

    assert s % self.chunked_one_step_attn_num_seq_split == 0
    w = s // self.chunked_one_step_attn_num_seq_split

    def _dynamic_qkv(query, key, value, atten_mask, start_chunk, num_chunks):
      key = jax.lax.dynamic_slice(
          key, [0, start_chunk, 0, 0], [b, num_chunks * w, n, h]
      )
      value = jax.lax.dynamic_slice(
          value, [0, start_chunk, 0, 0], [b, num_chunks * w, n, h]
      )
      atten_mask = jax.lax.dynamic_slice(
          atten_mask, [0, 0, start_chunk], [b, 1, num_chunks * w]
      )
      logits = self.qk_einsum('BNH,BSNH->BNS', query, key)
      if self.scale_logits_by_head_dims:
        logits = jnp.multiply(logits, 1.0 / np.sqrt(h))

      logits = self._cap_logits(logits)
      # Attention softmax is always carried out in fp32.
      logits = logits.astype(jnp.float32)
      # Apply attention masking
      padded_logits = py_utils.apply_mask_to_logits(logits, atten_mask)

      # Of shape [b, n, s]
      if self.attention_extra_logit is None:
        probs = jax.nn.softmax(padded_logits, axis=-1).astype(key.dtype)
      else:
        probs = jnp.exp(
            self._log_softmax_with_extra_logit(padded_logits)
        ).astype(key.dtype)
      return self.pv_einsum('BNS,BSNH->BNH', probs, value)

    dynamic_qkv_fns = []
    for c in range(self.chunked_one_step_attn_num_seq_split):
      dynamic_qkv_fns.append(functools.partial(_dynamic_qkv, num_chunks=c + 1))

    start_chunk = 0
    num_chunks = time_step // w - start_chunk + 1
    encoded = jax.lax.switch(
        num_chunks - 1,
        dynamic_qkv_fns,
        query,
        key,
        value,
        atten_mask,
        start_chunk,
    )

    if self.zero_fully_masked:
      # Return zeros for tokens which don't attend anything.
      fully_masked = jnp.all(
          atten_mask < py_utils.get_large_negative_number(jnp.float32) / 2,
          axis=-1,
      )[..., jnp.newaxis]
      encoded *= 1 - fully_masked

    encoded = self._shard_bnh(encoded)
    return encoded, jnp.zeros((0,))  # pytype: disable=bad-return-type  # jax-ndarray


class MXUDotProductAttention(layers.DotProductAttention):
  """DotProductAttention that uses MXU for Dot attention."""

  def _dot_atten_one_step(
      self,
      query: JTensor,
      key_state_name: str,
      value_state_name: str,
      atten_mask: JTensor,
      relative_bias: JTensor | None = None,
      time_step: JTensor | None = None,
  ) -> tuple[JTensor, JTensor]:
    """Dot attention function for queries with 1 time step.

    Args:
      query: JTensor of shape [B, N, H].
      key_state_name: Name of the decoding key state variable.
      value_state_name: Name of the decoding value state variable.
      atten_mask: JTensor of shape [1|B, 1, S] which is a mask that is applied
        to prevent attention between unwanted pairs. This has already been
        converted into large negative logits. The first dimension is allowed to
        be of size 1, if the mask is shared by all items in the batch (e.g.,
        only a causal mask).
      relative_bias: Relative bias of shape [1|B, N, 1, S].
      time_step: A scalar. The time step tensor.

    Returns:
      encoded: JTensor of shape [B, N, H].
      probs: JTensor of shape [B, N, S].
    """
    if relative_bias is not None:
      return super()._dot_atten_one_step(
          query, key_state_name, value_state_name,
          atten_mask, relative_bias, time_step)
    del time_step
    key = self._shard_blnh(self.get_decode_state(key_state_name))
    value = self._shard_blnh(self.get_decode_state(value_state_name))
    k_b = key.shape[0]
    q_b = query.shape[0]
    if q_b != k_b:
      if q_b % k_b != 0:
        raise ValueError(
            f'q batch size {q_b} is not divisible by state batch size {k_b}'
        )
      key = jnp.repeat(key, q_b // k_b, axis=0)
      value = jnp.repeat(value, q_b // k_b, axis=0)
    if atten_mask.shape[0] != q_b and atten_mask.shape[0] != 1:
      assert atten_mask.shape[0] == k_b, (atten_mask.shape, k_b)
      atten_mask = jnp.repeat(atten_mask, q_b // k_b, axis=0)
    # query is 3d.
    query = self._shard_bnh(query)

    b, s, n, h = key.shape
    base_layer.assert_has_shape(value, [b, s, n, h])
    base_layer.assert_has_shape(query, [b, n, h])
    base_layer.assert_has_shape(atten_mask, [-1, 1, s])
    query = self._scale_query(query)
    # Expand the query tensor from BNH to BNHD to force matrix-style
    # multiplication within qk_einsum and pv_einsum for better tpu performance.
    # This is still computationally efficient because query and logits tensors
    # are small relative to the key tensor, and these einsum operation's primary
    # bottleneck is HBM bandwidth rather than raw computation (flops).
    query = jnp.expand_dims(query, axis=-1)
    query = jnp.repeat(query, 2, axis=-1)
    logits = self.qk_einsum('BNHD,BSNH->BNDS', query, key)

    if self.scale_logits_by_head_dims:
      logits = jnp.multiply(logits, 1.0 / np.sqrt(h))

    logits = self._cap_logits(logits)
    # Attention softmax is always carried out in fp32.
    logits = logits.astype(jnp.float32)
    # Apply attention masking
    padded_logits = py_utils.apply_mask_to_logits(
        logits, jnp.expand_dims(atten_mask, axis=1))
    # Of shape [b, n, 1, s]
    if self.attention_extra_logit is None:
      probs = jax.nn.softmax(padded_logits, axis=-1).astype(key.dtype)
    else:
      probs = jnp.exp(self._log_softmax_with_extra_logit(padded_logits)).astype(
          key.dtype
      )
    # Compute the attention context.
    encoded = self.pv_einsum('BNDS,BSNH->BNDH', probs, value)
    # Get back to the original BNH tensor
    encoded = encoded[:, :, 0, :]
    if self.zero_fully_masked:
      # Return zeros for tokens which don't attend anything.
      fully_masked = jnp.all(
          atten_mask < py_utils.get_large_negative_number(jnp.float32) / 2,
          axis=-1,
      )[..., jnp.newaxis]
      encoded *= 1 - fully_masked

    encoded = self._shard_bnh(encoded)
    return encoded, probs

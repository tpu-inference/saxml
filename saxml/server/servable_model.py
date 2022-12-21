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
"""Wraps a model with service APIs."""

import abc
from typing import Any, Dict, List, Optional, Tuple

from saxml.server import servable_model_params

HostTensors = Any
DeviceTensors = Any
ExtraInput = Dict[str, float]


class ServableMethod(abc.ABC):
  """Base class for method implementation and its pre- and post-processing.

  Subclasses need to implement the abstract methods.
  """

  def __init__(self, method_params: servable_model_params.ServableMethodParams):
    self._sorted_batch_sizes = method_params.get_batch_size()
    if isinstance(self._sorted_batch_sizes, int):
      self._sorted_batch_sizes = [self._sorted_batch_sizes]
    assert isinstance(self._sorted_batch_sizes, list)
    self._sorted_batch_sizes = sorted(self._sorted_batch_sizes)
    self._max_live_batches = method_params.get_max_live_batches()
    self._extra_inputs = method_params.get_default_extra_inputs()

  @classmethod
  @abc.abstractmethod
  def service_id(cls) -> str:
    """Unique ID for the model service that supports this model."""

  @property
  def sorted_batch_sizes(self) -> List[int]:
    """A list of sorted supported (ascending order) batch sizes."""
    return self._sorted_batch_sizes

  @property
  def default_extra_inputs(self) -> Optional[ExtraInput]:
    """Default extra inputs for requests that do not specify them."""
    return self._extra_inputs

  @abc.abstractmethod
  def unload(self) -> None:
    """Clears references held by this method."""

  @abc.abstractmethod
  def input_to_device(self, one_core_inputs: HostTensors,
                      unpadded_batch_size: int) -> DeviceTensors:
    """Transfers input data to device. Pads incomplete batches."""

  @abc.abstractmethod
  def output_to_host(self,
                     output_tensors: DeviceTensors,
                     unpadded_batch_size: int) -> HostTensors:
    """Fetches device outputs to host. Removes batch padding."""

  @abc.abstractmethod
  def remove_batch_padding(self, host_tensors: HostTensors,
                           unpadded_batch_size: int) -> HostTensors:
    """Removes batch padding."""

  @property
  def batch_size(self) -> int:
    return self.sorted_batch_sizes[-1] if self.sorted_batch_sizes else 1

  @property
  def max_live_batches(self) -> int:
    """Maximum number of live batches in the server for this method."""
    return self._max_live_batches

  @abc.abstractmethod
  def pre_processing(self, raw_inputs: List[Any]) -> HostTensors:
    """Preprocesses an unpadded batch of data into host arrays."""

  @abc.abstractmethod
  def update_extra_inputs(
      self,
      input_batch: HostTensors,
      batch_size: int,
      extra_inputs: Optional[List[ExtraInput]] = None) -> HostTensors:
    """Updates mutable input keys to input batch.

    Users would like to update some input keys for the input batch through
    PRC requests. This function updates the per example mutable input value in
    the input batch from extra_inputs.

    Args:
      input_batch: Nested host arrays for device computation function input. It
        could be mutated.
      batch_size: Batch size of the input_batch.
      extra_inputs: Optional list of dictionary for {input_key: scalar_value}
        for each example. The keys in different elements of list could be
        different. The element in the list could be an empty dictionary. When it
        is None, when fill extra_inputs with self.default_extra_inputs.

    Returns:
      Updated input batch.
    """

  @abc.abstractmethod
  def post_processing(self, compute_outputs: HostTensors) -> List[Any]:
    """Postprocesses the output host arrays to final host output."""

  @abc.abstractmethod
  def device_compute(self, input_batch: DeviceTensors,
                     unpadded_batch_size: int) -> DeviceTensors:
    """Executes the device computation."""

  @property
  @abc.abstractmethod
  def streamable(self) -> bool:
    """Whether this method supports streaming."""

  @abc.abstractmethod
  def dequeue_stream_output(self) -> Tuple[HostTensors, bool]:
    """Dequeue streamed tensors. Blocking if empty."""

  @abc.abstractmethod
  def enqueue_stream_output(self, stream_outputs: HostTensors) -> None:
    """Enqueue streamed tensors from device."""

  def get_padded_batch_size(self, unpadded_batch_size: int) -> int:
    for bs in self.sorted_batch_sizes:
      if bs >= unpadded_batch_size:
        return bs
    raise ValueError(
        f'Batch size larger than maximum: {unpadded_batch_size} vs '
        f'{self.batch_size}')

  def compute(self,
              raw_inputs: List[Any],
              extra_inputs: Optional[List[ExtraInput]] = None) -> List[Any]:
    """Executes pre_processing, device_compute, and post_processing."""
    unpadded_batch_size = len(raw_inputs)
    if unpadded_batch_size > self.batch_size:
      raise ValueError('Inputs to compute() had a larger batch size ('
                       f'{unpadded_batch_size}) than was '
                       f'configured ({self.batch_size})')
    inputs = self.pre_processing(raw_inputs)
    inputs = self.update_extra_inputs(inputs, unpadded_batch_size, extra_inputs)
    inputs = self.input_to_device(inputs, unpadded_batch_size)
    outputs = self.device_compute(
        inputs, self.get_padded_batch_size(unpadded_batch_size))
    outputs = self.output_to_host(outputs, unpadded_batch_size)
    return self.post_processing(outputs)

  @abc.abstractmethod
  def compute_with_dummy_data(self, unpadded_batch_size: int) -> DeviceTensors:
    """Executes device computation with dummy inputs."""
    # This is needed for multi-host SPMD programs to execute in sync.


class ServableModel(abc.ABC):
  """Base class for service implementation, backed by a model."""

  def __init__(self):
    self._methods: Dict[str, ServableMethod] = {}
    self._acls: Dict[str, str] = {}

  @property
  def methods(self) -> Dict[str, ServableMethod]:
    return self._methods

  def method(self, method: str) -> ServableMethod:
    """Gets a method with the given name."""
    return self._methods[method]

  def unload(self) -> None:
    """Clears references held by this model."""
    for method in self._methods.values():
      method.unload()
    self._methods = {}

  def save(self, checkpoint_path: Optional[str]) -> None:
    raise NotImplementedError('Save model not implemented')

  def add_method(self, key: str, method: ServableMethod) -> None:
    """Adds an initialized method."""
    self._methods[key] = method

  def set_acls(self, acls: Dict[str, str]):
    """Sets the ACLs for this model.

    Args:
      acls: A dictionary from method names (e.g., lm.score) to the name of an
        access control list (e.g., sax-log-access-acl).
    """
    self._acls = acls

  def get_acl(self, method_name: str):
    """Returns the ACL name for the method name.

    Args:
      method_name: The method name (e.g., lm.score).

    Returns:
      None if no explicit ACL name is given. Otherwise, returns
      the ACL name (e.g., sax-log-access-acl).
    """
    return self._acls.get(method_name, None)

  @abc.abstractmethod
  def supports_dummy_compute_on_primary(self) -> bool:
    """Returns if compute_with_dummy_data() can be used by the primary host."""
    # This allows optimizations that performs mult-host sync before the
    # preprocessing, and if error occurred during preprocessing, dummy data can
    # be used to allow the primary host to execute the same program which was
    # already communicated to the secondary hosts.

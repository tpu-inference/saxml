"""Flags for tuning XLA for LLM inference."""

BASE_XLA_TPU_FLAGS = {
    'xla_jf_auto_cross_replica_sharding': 'False',
    'xla_tpu_nd_short_transfer_max_chunks': '2048',
    'xla_tpu_perform_spmd_cse_prevention': 'True',
    'xla_tpu_rwb_fusion': 'False',
}

DEFAULT_FLAGS = {
    'xla_tpu_autofdo': 'false',
    'xla_tpu_rwb_fusion': 'false',
    'xla_tpu_perform_spmd_cse_prevention': 'true',
    'xla_jf_auto_cross_replica_sharding': 'false',
}

MBLO_FLAGS = {
    'xla_tpu_enforce_prefetch_fifo_order': 'true',
    'xla_tpu_memory_bound_loop_optimizer_options': 'enabled:true',
}

DAO_FLAGS = {
    'xla_tpu_permute_size4_cross_module_rings': 'true',
}

CM_FLAGS = {
    'xla_jf_spmd_threshold_for_windowed_einsum_mib': '0',
    'xla_enable_async_collective_permute': 'true',
    'xla_tpu_spmd_unroll_windowed_einsum': 'true',
}

TUNED_FLAGS = {
    'xla_tpu_order_dot_after_layout': 'False',
    'xla_tpu_scoped_vmem_limit_kib': '28672',
    'xla_tpu_async_copy_bandwidth_scaling_factor': '0.22570640086881308',
}

STRENGTH_FLAGS = {
    'xla_tpu_enable_dot_strength_reduction': 'false',
}

DOT_DOT_FLAGS = {
    'xla_tpu_dot_dot_fusion_duplicated': 'true',
}

NEW_TED_REFINED_BS32_FLAGS = {
    'xla_tpu_dot_dot_fusion': 'False',
    'xla_tpu_enable_aggressive_broadcast_priority_update': 'True',
    'xla_tpu_enable_experimental_fusion_cost_model': 'True',
    'xla_tpu_enforce_prefetch_fifo_order': 'True',
    'xla_tpu_order_dot_after_layout': 'False',
    'xla_tpu_reduce_loop_fusion_dup_with_unfusable_user': 'True',
    'xla_tpu_scoped_vmem_limit_kib': '30720',
    'xla_tpu_use_repeated_instance_for_preferred_prefetch_time': 'True',
    'xla_vf_vmem_enable_cross_program_prefetch_freeing': 'False',
    'xla_vf_vmem_max_outstanding_evictions': '1',
}

NEW_TUNED_BS32_FLAGS = {
    'xla_jf_rematerialization_percent_shared_memory_limit': '98',
    'xla_tpu_async_copy_bandwidth_scaling_factor': '1.0171797766001474',
    'xla_tpu_copy_elision_analysis_allowance': '675',
    'xla_tpu_copy_fusion_pad_unpad_ratio': '183.24665260042565',
    'xla_tpu_copy_insertion_use_region_analysis_limit': '1308',
    'xla_tpu_dot_dot_fusion': 'False',
    'xla_tpu_enable_aggressive_broadcast_priority_update': 'True',
    'xla_tpu_enable_experimental_fusion_cost_model': 'True',
    'xla_tpu_enforce_prefetch_fifo_order': 'True',
    'xla_tpu_licm_size_inflation_ratio': '1.1254822383176215',
    'xla_tpu_msa_inefficient_use_to_copy_ratio': '0.7746306998901824',
    'xla_tpu_nd_short_transfer_max_chunks': '6369',
    'xla_tpu_order_dot_after_layout': 'False',
    'xla_tpu_prefetch_interval_picker_size_override': '405728676',
    'xla_tpu_reduce_loop_fusion_dup_with_unfusable_user': 'True',
    'xla_tpu_scoped_vmem_limit_kib': '30148',
    'xla_tpu_use_repeated_instance_for_preferred_prefetch_time': 'True',
    'xla_tpu_vector_load_fusion_window': '555',
    'xla_tpu_vector_store_fusion_window': '3495',
    'xla_vf_vmem_enable_cross_program_prefetch_freeing': 'False',
    'xla_vf_vmem_max_outstanding_evictions': '1',
    'xla_vf_vmem_max_outstanding_prefetches': '11',
    'xla_vf_vmem_max_overlap_to_mem_size_async_copy_ratio': '5.350891723154582',
    'xla_vf_vmem_max_repacks': '2',
    'xla_vf_vmem_max_retries': '7',
    'xla_vf_vmem_min_overlap_to_async_copy_ratio': '1.0951678793653812',
    'xla_vf_vmem_preferred_overlap_to_async_copy_ratio': (
        '0.0019865563814992176'
    ),
    'xla_tpu_memory_bound_loop_optimizer_options': 'enabled:true',
}

NEW_TUNED_BS16_FLAGS = {
    'xla_jf_conv_output_fusion': 'False',
    'xla_jf_rematerialization_percent_shared_memory_limit': '99',
    'xla_tpu_async_copy_bandwidth_scaling_factor': '1.496232791831925',
    'xla_tpu_copy_elision_analysis_allowance': '4',
    'xla_tpu_copy_fusion_pad_unpad_ratio': '1.5687246946851794',
    'xla_tpu_copy_insertion_use_region_analysis_limit': '6',
    'xla_tpu_dot_dot_fusion': 'False',
    'xla_tpu_dot_dot_fusion_duplicated': 'True',
    'xla_tpu_enforce_prefetch_fifo_order': 'True',
    'xla_tpu_licm_size_inflation_ratio': '0.8883449773946741',
    'xla_tpu_memory_bound_loop_optimizer_options': 'enabled:true',
    'xla_tpu_msa_inefficient_use_to_copy_ratio': '0.9876861835383646',
    'xla_tpu_nd_short_transfer_max_chunks': '4870',
    'xla_tpu_order_dot_after_layout': 'False',
    'xla_tpu_prefetch_interval_picker_size_override': '963616',
    'xla_tpu_scoped_vmem_limit_kib': '31686',
    'xla_tpu_vector_load_fusion_window': '2246',
    'xla_tpu_vector_store_fusion_window': '2998',
    'xla_vf_vmem_max_outstanding_evictions': '223',
    'xla_vf_vmem_max_outstanding_prefetches': '143',
    'xla_vf_vmem_max_overlap_to_mem_size_async_copy_ratio': (
        '0.004745847777308703'
    ),
    'xla_vf_vmem_max_repacks': '45',
    'xla_vf_vmem_max_retries': '7',
    'xla_vf_vmem_min_overlap_to_async_copy_ratio': '0.18832167995760815',
    'xla_vf_vmem_preferred_overlap_to_async_copy_ratio': '0.034587971107344614',
}

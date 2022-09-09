| Operation    | Kernel(s)                                                                                                                          |
| ------------ |
| addmm        | cutlass                                                                                                                            |
| gather       | _scatter_gather_element_wise_kernel                                                                                                |
| index_add_   | indexAddLargeIndex                                                                                                                 |
| index_select | indexSelectLargeIndex                                                                                                              |
| scatter_add  | _scatter_gather_element_wise_kernel                                                                                                |
| scatter_max  | scatter_kernel and scatter_arg_kernel                                                                                              |
| scatter_mean | _scatter_gather_element_wise_kernel                                                                                                |
| scatter_min  | scatter_kernel and scatter_arg_kernel                                                                                              |
| spmm         | csrmm_kernel and elementwise_kernel                                                                                                |
| spspmm       | _kernel_agent and DeviceSegmentedRadixSortKernel and binary_search_lb_offset_kernel and DeviceReduceByKeyKernel                    |
| transpose    | unrolled_elementwise_kernel   and reduce_kernel  and distribution_elementwise_grid_stide_kernel  and vectorized_elementwise_kernel |
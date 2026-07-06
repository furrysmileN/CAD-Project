"""ABCdataset 批量编排子模块。

提供解压、配对、阶段封装与共享工具，供顶层 `run_abc_batch.py` 串联使用。

本子模块不修改既有 `build_step_assets.py` / `build_occlusion_assets.py` /
`describe_step_with_qwen.py` / `batch_describe_cad_with_qwen.py` /
`clean_cad_contexts.py` 的默认行为；所有跨阶段约束（路径布局、日志格式、
Qwen API 参数）都集中在本子模块下。
"""

__all__ = [
    "paths",
    "logging_utils",
    "extract_archives",
    "pair_samples",
    "make_batches",
    "stage_batch_inputs",
    "stage_assets",
    "stage_occlusion",
    "stage_describe",
    "archive_batch",
    "global_index",
    "cleanup_report",
    "post_archive_describe",
]

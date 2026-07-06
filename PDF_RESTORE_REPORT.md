# PDF Restore Report

- Source PDF: `/root/autodl-tmp/代码.pdf`
- Restored code page range: `1-336`
- Ignored pages: `337-394` (实验报告、语雀示例/帮助内容，按用户要求不还原为项目代码)
- Restored files: `41`

## Files

- `modified_cadrille/filter.py`: pages 6-10, 180 lines, ok
- `modified_cadrille/train_qwen.py`: pages 12-25, 602 lines, ok
- `modified_cadrille/configs-qwn3_train.yaml`: pages 28-29, 68 lines, ok
- `cad_data_gen/train_qwen3vl_point_finetune.py`: pages 32-47, 602 lines, ok
- `cad_data_gen/step_mesh_backends.py`: pages 49-55, 292 lines, ok
- `cad_data_gen/step_assets.py`: pages 57-58, 64 lines, ok
- `cad_data_gen/qwen3vl_sync_baseline.py`: pages 60-62, 116 lines, ok
- `cad_data_gen/qwen3vl_point_ablation.py`: pages 64-66, 104 lines, ok
- `cad_data_gen/run_abc_batch.py`: pages 68-80, 542 lines, ok
- `cad_data_gen/render_step_with_blender.py`: pages 82-95, 582 lines, ok
- `cad_data_gen/pointcloud.py`: pages 96-96, 20 lines, ok
- `cad_data_gen/pipeline_state.py`: pages 98-99, 86 lines, ok
- `cad_data_gen/clean_cad_contexts.py`: pages 101-106, 240 lines, ok
- `cad_data_gen/cad_code_to_mesh.py`: pages 109-110, 51 lines, ok
- `cad_data_gen/cad_context_cleaner.py`: pages 112-121, 379 lines, page 121: appended unnumbered continuation
- `cad_data_gen/build_step_assets.py`: pages 123-136, 602 lines, ok
- `cad_data_gen/build_residual_step_files.py`: pages 138-141, 158 lines, ok
- `cad_data_gen/build_occlusion_assets.py`: pages 143-156, 602 lines, ok
- `cad_data_gen/abc_batch/stage_occlusion.py`: pages 158-164, 278 lines, ok
- `cad_data_gen/abc_batch/stage_describe.py`: pages 166-174, 385 lines, ok
- `cad_data_gen/abc_batch/stage_batch_inputs.py`: pages 176-185, 418 lines, ok
- `cad_data_gen/abc_batch/stage_assets.py`: pages 187-200, 602 lines, ok
- `cad_data_gen/abc_batch/post_archive_describe.py`: pages 202-207, 255 lines, ok
- `cad_data_gen/abc_batch/paths.py`: pages 209-213, 203 lines, ok
- `cad_data_gen/abc_batch/pair_samples.py`: pages 215-224, 438 lines, ok
- `cad_data_gen/abc_batch/make_batches.py`: pages 226-232, 293 lines, ok
- `cad_data_gen/abc_batch/logging_utils.py`: pages 234-236, 115 lines, ok
- `cad_data_gen/abc_batch/global_index.py`: pages 238-243, 261 lines, ok
- `cad_data_gen/abc_batch/follow_describe_assets.py`: pages 246-260, 602 lines, ok
- `cad_data_gen/abc_batch/extract_archives.py`: pages 262-273, 535 lines, ok
- `cad_data_gen/abc_batch/cleanup_report.py`: pages 275-279, 222 lines, ok
- `cad_data_gen/abc_batch/archive_batch.py`: pages 282-289, 360 lines, ok
- `cad_data_gen/abc_batch/__init__.py`: pages 291-291, 25 lines, ok
- `cad_data_gen/encoder/training.py`: pages 294-301, 338 lines, ok
- `cad_data_gen/encoder/preprocess.py`: pages 304-309, 233 lines, ok
- `cad_data_gen/encoder/logging_utils.py`: pages 311-312, 45 lines, ok
- `cad_data_gen/encoder/core.py`: pages 314-319, 229 lines, ok
- `cad_data_gen/encoder/config.py`: pages 321-323, 105 lines, ok
- `cad_data_gen/encoder/batch.py`: pages 325-330, 243 lines, ok
- `cad_data_gen/encoder/__init__.py`: pages 332-333, 76 lines, ok
- `cad_data_gen/configs/abc_5k_generation_requirements.yaml`: pages 335-336, 58 lines, ok

## Notes

- Code was reconstructed from PDF text blocks and line-number alignment. Visual line wraps were joined back into source lines by page line numbers.
- Pages `337-394` were intentionally ignored as non-project-code content.
- Locations marked with notes should be manually compared with the PDF if exact byte-for-byte fidelity is required.

## Validation Summary

- Python files checked: `39`; passed: `25`; flagged: `14`.
- YAML files checked: `2`; errors: `0`.
- Detailed validation report: `PDF_RESTORE_VALIDATION.md`.

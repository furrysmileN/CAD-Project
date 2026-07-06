# PDF Restore Validation

- Python files checked: `39`
- Python files passed `py_compile`: `25`
- Python files needing manual PDF复核: `14`
- YAML files checked: `2`
- YAML errors: `0`
- Ignored PDF pages: `337-394` (非项目代码内容)

## Python Syntax Issues

- `cad_data_gen/abc_batch/follow_describe_assets.py`:   File "/root/autodl-tmp/cad_data_gen/abc_batch/follow_describe_assets.py", line 220 |     nce(row.get("conversion_metadata"), dict) else None,mesh_metrics=row.get("mesh_metrics") if isinstance(row.get("m |                                                                                                                    ^ | SyntaxError: unterminated string literal (detected at line 220) | 
- `cad_data_gen/abc_batch/make_batches.py`:   File "/root/autodl-tmp/cad_data_gen/abc_batch/make_batches.py", line 182 |     -max-input-bytes") |                     ^ | SyntaxError: unterminated string literal (detected at line 182) | 
- `cad_data_gen/abc_batch/stage_assets.py`:   File "/root/autodl-tmp/cad_data_gen/abc_batch/stage_assets.py", line 602 |     return res |               ^ | SyntaxError: expected 'except' or 'finally' block | 
- `cad_data_gen/abc_batch/stage_describe.py`:   File "/root/autodl-tmp/cad_data_gen/abc_batch/stage_describe.py", line 228 |     t}" |      ^ | SyntaxError: unmatched '}' | 
- `cad_data_gen/build_occlusion_assets.py`:   File "/root/autodl-tmp/cad_data_gen/build_occlusion_assets.py", line 600 |     def _apply_mask_visualization( |                                  ^ | SyntaxError: '(' was never closed | 
- `cad_data_gen/build_step_assets.py`:   File "/root/autodl-tmp/cad_data_gen/build_step_assets.py", line 487 |     2)): |      ^ | SyntaxError: unmatched ')' | 
- `cad_data_gen/cad_context_cleaner.py`: Sorry: IndentationError: unexpected indent (cad_context_cleaner.py, line 139)
- `cad_data_gen/clean_cad_contexts.py`:   File "/root/autodl-tmp/cad_data_gen/clean_cad_contexts.py", line 87 |     tures=args.raw_max_ofs_features) |                                    ^ | SyntaxError: unmatched ')' | 
- `cad_data_gen/encoder/core.py`:   File "/root/autodl-tmp/cad_data_gen/encoder/core.py", line 130 |     nts)) |        ^ | SyntaxError: unmatched ')' | 
- `cad_data_gen/render_step_with_blender.py`:   File "/root/autodl-tmp/cad_data_gen/render_step_with_blender.py", line 167 |     Any) -> None: |        ^ | SyntaxError: unmatched ')' | 
- `cad_data_gen/run_abc_batch.py`:   File "/root/autodl-tmp/cad_data_gen/run_abc_batch.py", line 214 |     stderr) |           ^ | SyntaxError: unmatched ')' | 
- `cad_data_gen/step_mesh_backends.py`:   File "/root/autodl-tmp/cad_data_gen/step_mesh_backends.py", line 212 |     ode {exc.returncode}" |                         ^ | SyntaxError: unterminated string literal (detected at line 212) | 
- `cad_data_gen/train_qwen3vl_point_finetune.py`:   File "/root/autodl-tmp/cad_data_gen/train_qwen3vl_point_finetune.py", line 134 |     ize)logger = JsonlLogger(output_dir / f"train_log_rank{dist_ctx.rank}.jsonl" if not dist_ctx.is_main else output_dir / "train_log.jsonl", enabled= |        ^ | SyntaxError: unmatched ')' | 
- `modified_cadrille/train_qwen.py`:   File "/root/autodl-tmp/modified_cadrille/train_qwen.py", line 409 |     essage)s") |           ^ | SyntaxError: unmatched ')' | 

## Notes

- These issues are caused by PDF visual wrapping or apparent truncation, not by ignored tail pages.
- `cad_data_gen/build_occlusion_assets.py` appears truncated at the end of the project-code section in the PDF.

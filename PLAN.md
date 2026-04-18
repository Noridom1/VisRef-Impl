# Reproduce VisRef on InternVL3.5-8B + MathVista

## Summary
- The current repo is a scaffold, not a reproducer yet. The core VisRef/ST/TSR control flow exists, but the model-side behavior in [models/internvl.py](</d:/projects/VisRes-Impl/models/internvl.py:184>) is still placeholder/random: visual token extraction, reasoning-step generation, reasoning embeddings, answer distribution, and final answer.
- The method runners ([methods/visref.py](</d:/projects/VisRes-Impl/methods/visref.py:16>), [methods/st.py](</d:/projects/VisRes-Impl/methods/st.py:15>), [methods/tsr.py](</d:/projects/VisRes-Impl/methods/tsr.py:17>)) do not pass answer choices into the reasoning loop, so benchmark behavior cannot be paper-faithful yet.
- Evaluation and experiment orchestration are incomplete: current scoring is simple normalized exact match only ([eval/metrics.py](</d:/projects/VisRes-Impl/eval/metrics.py:6>), [engine/runner.py](</d:/projects/VisRes-Impl/engine/runner.py:28>)); ablation overrides and parallel-budget aggregation are still placeholders ([scripts/run_ablation.py](</d:/projects/VisRes-Impl/scripts/run_ablation.py:52>), [scripts/run_parallel_budget.py](</d:/projects/VisRes-Impl/scripts/run_parallel_budget.py:39>)).
- The environment is not reproduction-ready yet: there is no dataset payload under `data/`, and the current Python env is missing `torch`.

## Key Changes
- Replace the placeholder InternVL reasoning state with a real stateful wrapper matching the paper loop from [arXiv:2603.00207](https://doi.org/10.48550/arXiv.2603.00207): generate one reasoning step `z_k`, extract its hidden states, build `M_k = Σ z z^T`, compute the DPP kernel `L_k(v_i,v_j)=v_i^T M_k v_j`, greedily select `m=floor(0.3|V|)` visual tokens, reinject them, and stop when predictive entropy drops below `0.25` or `k=10`.
- Keep the current CLI stable, but refactor the internal wrapper interface to be choice-aware and stateful. Required internal primitives:
  - `start_reasoning(question, image, choices, prompt_cfg)`
  - `generate_reasoning_step(state, extra_visual_tokens, reflection_instruction)`
  - `extract_reasoning_embeddings(state)`
  - `estimate_answer_distribution(state, choices)`
  - `generate_final_answer(state, choices)`
- Implement phase one only for `InternVL3.5-8B + MathVista`. Qwen and the full paper matrix should remain behind the same interface, but they are phase-two work.
- Make ST and TSR use the same decoding backend as VisRef, differing only in whether extra visual tokens are reinjected and whether the reflection prompt is added.
- Replace placeholder evaluation with MathVista-first answer extraction/normalization: parse structured `<answer>...</answer>` when present, fall back to robust text extraction, then compare with benchmark-aware normalization instead of raw lowercase string equality.
- Finish the experiment layer after single-sample bring-up:
  - `run_eval.py`: stable per-sample outputs for ST, TSR, VisRef
  - `run_ablation.py`: real overrides for `entropy_threshold` and `token_budget_ratio`
  - `run_parallel_budget.py`: multi-chain aggregation and fixed-budget reporting

## Test Plan
- Smoke test the environment first: model loads, one MathVista sample image/token pass succeeds, and there are no prompt/image token count mismatches.
- Unit test the VisRef math on synthetic arrays: `build_Mk`, kernel construction, greedy DPP selection uniqueness, and budget enforcement.
- Integration test one MathVista sample through ST, TSR, and VisRef:
  - choices survive end-to-end
  - VisRef records non-empty `selected_token_counts`
  - stopping occurs by entropy threshold or `K_max`
  - final answer is not a placeholder string
- Run a small MathVista slice first, inspect failure modes, then scale to full `testmini` for the first milestone report.
- Acceptance for phase one: `InternVL3.5-8B + MathVista` runs end-to-end with paper defaults, produces saved summaries/records, and shows meaningful separation between ST, TSR, and VisRef on a held-out slice before full-benchmark execution.

## Assumptions
- First milestone is `InternVL3.5-8B + MathVista`; Qwen, MM-Star, and MathVision come after the first stable reproduction path.
- We are using a hybrid strategy: get a real runnable implementation first, but keep the VisRef loop paper-faithful rather than shipping another approximation.
- Paper defaults for phase one are fixed to `delta_entropy=0.25`, `token_budget_ratio=0.3`, and `K_max=10`, following the paper’s implementation details and appendix ([ResearchGate mirror](https://www.researchgate.net/publication/401468922_VisRef_Visual_Refocusing_while_Thinking_Improves_Test-Time_Scaling_in_Multi-Modal_Large_Reasoning_Models)).
- Reproduction depends on a CUDA environment with `torch` installed and access to the InternVL checkpoint; until that is in place, we can implement and dry-run structure, but not verify benchmark results.

# cascade-mind — Remaining Work

## 🚨 BLOCKING — Automated Checks (will fail validation if not done)

These are hard gates. If any is missing, the submission won't reach a human judge.

- [ ] **Fix training notebook** (`grpo_sre_training_8b_final.ipynb`) — critical bug: `trainer.train()` is never called; the current loop runs rollouts and overwrites log history with a synthetic floor+ramp signal. No actual gradient updates happen. Fix this before the July 25 compute run.
- [ ] **Commit real reward curve PNG to repo** — `assets/reward_curve.png` (or `assets/grpo_learning_curves_8b.png`) must be a PNG committed to the repo from a real training run. Wandb-only or Colab-cell-only plots don't count for validation.
- [ ] **Commit real loss curve PNG to repo** — validation checks for at minimum loss + reward plots as `.png`/`.jpg` files in the repo.
- [ ] **Add Results section to README** with before/after table + embedded reward curve image — README must link every deliverable or validation treats it as missing.
- [ ] **Add video badge + blog badge to README** — once recorded/published; validation checks README links.

## ⚡ HIGH IMPACT — Storytelling (30% of judging score)

- [ ] Record 3-min raw playground walkthrough (seed=42, difficulty=medium)
- [ ] Speed up to 1.5 min in iMovie / DaVinci
- [ ] Record voiceover separately, layer in
- [ ] Upload to YouTube → get public link
- [ ] Write and publish HF blog post (links: Space + GitHub + API docs + Colab; cover Brier novelty + GRPO results)

## Blocked on Demo Video (do in order)

- [ ] Record 3-min raw playground walkthrough (seed=42, difficulty=medium)
- [ ] Speed up to 1.5 min in iMovie / DaVinci
- [ ] Record voiceover separately, layer in
- [ ] Upload to YouTube → get public link

## Blocked on Training Run (July 25 compute day)

- [ ] Create clean training notebook (`notebooks/grpo_sre_training_final.ipynb`)
  - Start from `grpo_sre_training_8b_final.ipynb` as base
  - Make sure reward curve is logged and exportable as PNG
- [ ] Run GRPO training on 8B model, capture reward curve
- [ ] Generate `assets/reward_curve.png` from training logs
- [ ] Record baseline numbers per difficulty for results table:
  - Easy: 0.81 (confirmed)
  - Medium: 0.61 (confirmed)
  - Hard: 0.38 (confirmed)
  - After GRPO: TBD

## README Updates (after video + training)

- [ ] Add video badge + link at top badges row
- [ ] Add Results section with before/after table + reward curve image
  - Embed `assets/reward_curve.png` via raw GitHub URL
  - Fill in GRPO numbers once training is done
- Include all links HF space, HF docs, youtube video link, repository link, HF blog, architectures etc.
- Collab links 
- add a section why this domain is research oriented
    - Compare a trained agent vs. a random/untrained baseline; quantitative and/or qualitative

## Blog Post (HuggingFace)

- [ ] Write blog post on HF
  - Link: https://huggingface.co/blog (publish under your HF account)
  - Cover: problem framing, architecture, Brier score novelty, GRPO results
  - Add Colab badge linking to training notebook
  - Link to Space + GitHub + API docs
- [ ] Add blog link to README badges row after publishing

## Quick Fixes (< 5 min each)

- [ ] `service_impact_environment.py` line 183: `version: str = "0.2.0"` → `"0.3.0"` (mismatches app.py)

## Done ✓

- [x] OpenEnv Rubric integration (FBetaRubric 80% + BrierScoreRubric 20%)
- [x] Brier score calibration reward
- [x] Supply chain domain (DomainConfig plugin)
- [x] World Modeling Layer v3 (BeliefTracker + ContradictionEngine + GraphPrior)
- [x] MCP endpoint (GET /mcp/manifest + POST /mcp JSON-RPC)
- [x] README full rewrite with architecture diagrams
- [x] API docs updated to v0.3.0 (domain-agnostic description, full tool table)
- [x] openenv.yaml updated with new tags and description
- [x] Architecture PNG assets generated and hosted on GitHub
- [x] Playground score card (F-beta + Budget + Brier tiles)
- [x] Trajectory Replay tab
- [x] GRPO export pipeline (/export/grpo endpoint)
- [x] Slides (Problem + Solution) updated for video

"""Corrida final: 6 alvos, seed Gemini one-shot + refinamento render-guided, 3 seeds de otimizador."""
import time
import tp2

critic = tp2.GeminiCritic()
approach = tp2.refine_approach(critic=critic, beam=3, n_proposals=2, rounds=2, verbose=True)

t0 = time.time()
run_dir, rows, summary = tp2.run_pipeline(
    approach,
    targets=None,                 # todos os 6 alvos
    k=3,
    optimiser_seeds=(0, 1, 2),    # 3 repeticoes (estocasticidade = Gemini)
    identity="final_3seeds",
)
print("RUN_DIR:", run_dir)
print("TEMPO_MIN:", round((time.time() - t0) / 60, 1))
print("\nTOP-3 por alvo:")
best = summary.get("best_per_target", summary)
for r in sorted(rows, key=lambda x: (x["target_name"], x["rank"])):
    if r["rank"] <= 3:
        print(f"  [{r['target_name']}] #{r['rank']} score={r['score']:.3f} "
              f"clip={r['clip_sim']:.3f} lpips={r['lpips']:.3f} rmse={r['rmse']:.3f} "
              f":: {r['prompt'][:70]}")

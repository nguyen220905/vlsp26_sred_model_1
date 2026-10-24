"""Pick the epoch count of the final all-data run from dev-A pooled EER (AS-norm), the plan's selection metric:
4 epochs only if kd_M_b6ft_proj_4ep beats the mean of the two 2-epoch seeds by more than the seed spread (0.02)."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..")  # the distill folder
g = lambda d: json.load(open(os.path.join(R, d, "results.json")))["devA_pooled"]["asnorm"]["eer"]  # noqa
two = [g("kd_M_b6ft_proj"), g("kd_M_b6ft_proj_seed1")]
four = g("kd_M_b6ft_proj_4ep")
mean2 = sum(two) / len(two)
epochs = 4 if four < mean2 - 0.02 else 2
dec = {"two_epoch_runs": two, "two_epoch_mean": round(mean2, 3), "four_epoch": four, "margin": 0.02, "epochs": epochs}
with open(os.path.join(HERE, "M_proj_alldata", "epoch_decision.json"), "w") as f:
    json.dump(dec, f, indent=1)
print(json.dumps(dec), file=sys.stderr)
print(epochs)

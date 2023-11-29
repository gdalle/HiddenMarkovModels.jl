using HiddenMarkovModels: LightCategorical
using Test

p = rand_prob_vec(10)
dist = LightCategorical(p)

x = [rand(dist) for _ in 1:1_000_000]
val_count = zeros(Int, length(p))
for k in x
    val_count[k] += 1
end
@test val_count ./ length(x) ≈ p atol=1e-2

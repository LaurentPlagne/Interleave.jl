# Noyaux de référence, partagés par la suite de tests et le banc de mesure.
#
# Ils sont écrits en notation scalaire ordinaire et ne portent AUCUNE annotation de
# vectorisation. Ils tournent tels quels avec T = Float32 (scalaire) ou T = Vec{P,Float32}
# (vectoriel) : c'est exactement la propriété que la bibliothèque doit garantir.
#
# ⚠️ Pas de `@fastmath` ici : la contraction FMA casserait la bit-exactitude
# scalaire ↔ vectorisé (AGENTS.md, invariant 1).

"""Balayage de Thomas pour un système tridiagonal (D diagonale, U sur-, L sous-diagonale)."""
function thomas!(X, D, U, L, B, S)
    @inbounds begin
        s = D[1]
        sm1 = inv(s)
        X[1] = B[1] * sm1
        for i in 2:length(X)
            S[i] = U[i-1] * sm1
            s    = D[i] - L[i] * S[i]
            X[i] = B[i] - L[i] * X[i-1]
            sm1  = inv(s)
            X[i] *= sm1
        end
        for i in length(X)-1:-1:1
            X[i] -= S[i+1] * X[i+1]
        end
    end
    X
end

"""Produit tridiagonal `R .= T*X`, pour le contrôle de résidu (invariant inverse)."""
function tridiag_mul!(R, D, U, L, X)
    n = length(X)
    @inbounds begin
        R[1] = D[1] * X[1] + U[1] * X[2]
        for i in 2:n-1
            R[i] = L[i] * X[i-1] + D[i] * X[i] + U[i] * X[i+1]
        end
        R[n] = L[n] * X[n-1] + D[n] * X[n]
    end
    R
end

"""Squared norm of one independent instance, written to its one-element output."""
function batch_squarednorm!(out, A)
    E = eltype(A)
    acc = zero(E)
    @inbounds for i in eachindex(A)
        x = A[i]
        acc += x * x
    end
    out[1] = acc
    out
end

"""Dot product of two independent instances, written to a one-element output."""
function batch_dot!(out, A, B)
    E = eltype(A)
    acc = zero(E)
    @inbounds for i in eachindex(A)
        acc += A[i] * B[i]
    end
    out[1] = acc
    out
end

"""
Filtre IIR biquad, forme directe I :
`y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]`.
Récurrence stricte sur `n` : aucun compilateur ne la vectorise le long du temps.
"""
function biquad!(Y, X, coeffs)
    b0, b1, b2, a1, a2 = coeffs
    T = eltype(Y)
    z = zero(T)
    x1 = z; x2 = z; y1 = z; y2 = z
    @inbounds for n in eachindex(Y)
        x0 = X[n]
        y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        Y[n] = y0
        x2 = x1; x1 = x0
        y2 = y1; y1 = y0
    end
    Y
end

# ---------------------------------------------------------------------------------
# Instances 2D : le lot porte des cartes (H, W), pas des vecteurs.
# `instance(A, k)` rend alors une vue 2D — le noyau s'écrit en indices (i, j).
# Parcours colonne-major (i contigu à l'intérieur), contrairement au C++ qui aplatit
# en row-major : c'est le même calcul, dans l'ordre que veut Julia.

"""
Convolution depthwise 3×3 (primitive de MobileNet / ConvNeXt).
`w` est le noyau 3×3 donné en `NTuple{9}`, ligne par ligne.
"""
function depthwise3x3!(out, inp, w)
    w00, w01, w02, w10, w11, w12, w20, w21, w22 = w
    H, W = size(inp)
    @inbounds for j in 2:W-1, i in 2:H-1
        out[i, j] = w00 * inp[i-1, j-1] + w01 * inp[i-1, j] + w02 * inp[i-1, j+1] +
                    w10 * inp[i,   j-1] + w11 * inp[i,   j] + w12 * inp[i,   j+1] +
                    w20 * inp[i+1, j-1] + w21 * inp[i+1, j] + w22 * inp[i+1, j+1]
    end
    out
end

"""
Pipeline vidéo : gradient de Sobel spatial fusionné avec une différence temporelle
au carré. `ab = (α, β)` pondère les deux réponses.
"""
function sobel_motion!(out, curr, prev, ab)
    α, β = ab
    eighth = lanetype(eltype(curr))(0.125)
    H, W = size(curr)
    @inbounds for j in 2:W-1, i in 2:H-1
        c00, c01, c02 = curr[i-1, j-1], curr[i-1, j], curr[i-1, j+1]
        c10, c11, c12 = curr[i,   j-1], curr[i,   j], curr[i,   j+1]
        c20, c21, c22 = curr[i+1, j-1], curr[i+1, j], curr[i+1, j+1]
        # `x + x` plutôt que `2x` : reste générique quel que soit le type d'élément.
        gx = (c02 + (c12 + c12) + c22) - (c00 + (c10 + c10) + c20)
        gy = (c20 + (c21 + c21) + c22) - (c00 + (c01 + c01) + c02)
        edge = (gx * gx + gy * gy) * eighth
        d = c11 - prev[i, j]
        out[i, j] = α * edge + β * (d * d)
    end
    out
end

"""
Black-Scholes 1D par différences finies, schéma de Crank-Nicolson : à chaque pas de
temps, un produit tridiagonal explicite puis une résolution de Thomas implicite.
Double récurrence (temps × balayage), donc doublement hostile à l'auto-vectorisation.
"""
function blackscholes_cn!(V, D, U, L, RHS, S, nt)
    N = length(V)
    E = eltype(V)
    two = one(E) + one(E)
    @inbounds for _ in 1:nt
        for i in 2:N-1
            RHS[i] = -L[i] * V[i-1] + (two - D[i]) * V[i] - U[i] * V[i+1]
        end
        RHS[1] = V[1]
        RHS[N] = V[N]
        s   = D[1]
        sm1 = inv(s)
        V[1] = RHS[1] * sm1
        for i in 2:N
            S[i] = U[i-1] * sm1
            s    = D[i] - L[i] * S[i]
            V[i] = RHS[i] - L[i] * V[i-1]
            sm1  = inv(s)
            V[i] *= sm1
        end
        for i in N-1:-1:1
            V[i] -= S[i+1] * V[i+1]
        end
    end
    V
end

"""
Laplacien à 7 points sur un volume 3-D. Écrit en indices naturels `[i, j, k]` : le noyau
ignore complètement l'entrelacement sous-jacent.
"""
function laplacien3d!(out, inp)
    n1, n2, n3 = size(inp)
    @inbounds for k in 2:n3-1, j in 2:n2-1, i in 2:n1-1
        out[i, j, k] = inp[i-1, j, k] + inp[i+1, j, k] +
                       inp[i, j-1, k] + inp[i, j+1, k] +
                       inp[i, j, k-1] + inp[i, j, k+1] - (inp[i, j, k] * 6)
    end
    out
end

"""
Balayage de Thomas le long de la **première** dimension d'une instance 2-D `(n, m)` :
`m` systèmes tridiagonaux de taille `n` par instance. Sert aux schémas ADI, où l'axe
paqueté est une dimension de la grille.
"""
function thomas_lines!(X, D, U, L, B, S)
    n, m = size(X)
    @inbounds for c in 1:m
        s = D[1, c]
        sm1 = inv(s)
        X[1, c] = B[1, c] * sm1
        for i in 2:n
            S[i] = U[i-1, c] * sm1
            s    = D[i, c] - L[i, c] * S[i]
            X[i, c] = B[i, c] - L[i, c] * X[i-1, c]
            sm1  = inv(s)
            X[i, c] *= sm1
        end
        for i in n-1:-1:1
            X[i, c] -= S[i+1] * X[i+1, c]
        end
    end
    X
end

# GPU test entry points.  KernelAbstractions can inline a named, concrete callable on
# every backend; closures capturing tuples or integers are deliberately kept out of the
# device launch path.  These wrappers still call the very same scalar kernels above, so
# the GPU suite exercises source reuse rather than a second implementation.
const GPU_BIQUAD_COEFFS = (0.2f0, 0.4f0, 0.2f0, -0.3f0, 0.1f0)
const GPU_DEPTHWISE_WEIGHTS = Float32.((1, 2, 1, 2, 4, 2, 1, 2, 1) ./ 16)
const GPU_SOBEL_WEIGHTS = (0.7f0, 0.3f0)
const GPU_BLACKSCHOLES_NT = 8

@inline gpu_biquad!(Y, X) = biquad!(Y, X, GPU_BIQUAD_COEFFS)
@inline gpu_depthwise3x3!(out, inp) = depthwise3x3!(out, inp, GPU_DEPTHWISE_WEIGHTS)
@inline gpu_sobel_motion!(out, curr, prev) = sobel_motion!(out, curr, prev, GPU_SOBEL_WEIGHTS)
@inline gpu_blackscholes_cn!(V, D, U, L, RHS, S) =
    blackscholes_cn!(V, D, U, L, RHS, S, GPU_BLACKSCHOLES_NT)

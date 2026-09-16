# Génère la démo audio de la page d'accueil, avec le noyau `biquad!` du dépôt.
#
#   julia --project=docs docs/media/make_biquad_demo.jl
#
# Legolas++ embarque un `biquad_demo.mp3` équivalent. Plutôt que de le recopier, on le
# REFABRIQUE ici : le son que l'on entend est alors produit par `apply!(biquad!, …)`,
# c'est-à-dire par le noyau même que la suite de tests valide bit à bit. Une démo qui
# viendrait d'une autre implémentation ne prouverait rien sur celle-ci.
#
# Le matériau est synthétique — un accord de synthé — donc aucune source tierce n'est
# nécessaire et la démo est entièrement reproductible.
#
# Chaque **canal du lot** porte une partielle de l'accord : les 64 canaux passent ensemble
# dans le driver, et le son restitué est leur somme. Le lot n'est donc pas décoratif, c'est
# bien le chemin DLI qui produit l'audio.

using Interleave
using Printf: @sprintf

include(joinpath(@__DIR__, "..", "..", "test", "kernels.jl"))

const T   = Float32
const FS  = 44_100          # Hz
const DUR = 3.0             # secondes par moitié
const NCH = 64              # canaux indépendants, comme la démo C++
const FC  = 600.0           # coupure du passe-bas, Hz
const Q   = 0.7071          # Butterworth

"""Passe-bas biquad RBJ, normalisé pour la convention du noyau :
`y = b0·x + b1·x₁ + b2·x₂ − a1·y₁ − a2·y₂`."""
function lowpass(fc, fs, q)
    ω = 2π * fc / fs
    α = sin(ω) / (2q)
    c = cos(ω)
    b0, b1, b2 = (1 - c) / 2, 1 - c, (1 - c) / 2
    a0, a1, a2 = 1 + α, -2c, 1 - α
    (T(b0 / a0), T(b1 / a0), T(b2 / a0), T(a1 / a0), T(a2 / a0))
end

"""Un accord mineur septième : chaque canal est une partielle d'une des quatre notes.

Peu de notes et BEAUCOUP de partielles, avec une décroissance en 1/h — un spectre de dent de
scie. Une première version répartissait 64 canaux sur 8 notes, ne montait qu'à 8 partielles et
plaçait donc presque toute l'énergie **sous** la coupure : le passe-bas devenait inaudible.
Une démo qu'on n'entend pas ne démontre rien."""
function chord(nch, nsamples, fs)
    semitones = (0, 3, 7, 10)                           # Am7
    root = 110.0                                        # A2
    X = zeros(T, nch, nsamples)
    for c in 1:nch
        note = semitones[mod1(c, length(semitones))]
        harm = cld(c, length(semitones))                # 1 … 16 partielles
        f = root * 2^(note / 12) * harm
        f >= fs / 2 && continue                          # pas de repliement
        detune = 1 + 0.0007 * sinpi(c / 7)               # léger chorus
        amp = T(0.55 / harm)
        phase = 2π * (c / nch)
        @inbounds for n in 1:nsamples
            X[c, n] = amp * sin(2π * f * detune * (n - 1) / fs + phase)
        end
    end
    X
end

"""Enveloppe douce : sans elle, les coupures franches claquent à l'écoute."""
function envelope!(v, fs)
    n = length(v)
    ramp = round(Int, 0.02fs)
    @inbounds for i in 1:min(ramp, n)
        g = T(i / ramp)
        v[i] *= g
        v[n - i + 1] *= g
    end
    v
end

function main()
    nsamples = round(Int, DUR * FS)
    coeffs = lowpass(FC, FS, Q)
    @info "coefficients" b0=coeffs[1] b1=coeffs[2] b2=coeffs[3] a1=coeffs[4] a2=coeffs[5]

    X = chord(NCH, nsamples, FS)

    # Le lot passe par le driver DLI, en paquets de 8.
    Xi = Interleave.Array{T,2,8}(X)
    Yi = similar(Xi)
    fill!(Yi, 0)
    apply!((y, x) -> biquad!(y, x, coeffs), Yi, Xi)

    # Contrôle : le résultat doit être bit à bit celui du noyau scalaire.
    Yref = zeros(T, NCH, nsamples)
    apply!((y, x) -> biquad!(y, x, coeffs), Yref, X)
    err = maximum(abs(Yi[c, n] - Yref[c, n]) for c in 1:NCH, n in 1:nsamples)
    err == 0 || error("DLI et scalaire divergent de $err — la démo ne serait pas honnête")
    @info "accord DLI ↔ scalaire exact"

    raw = vec(sum(X; dims = 1))
    filt = zeros(Float64, nsamples)
    @inbounds for c in 1:NCH, n in 1:nsamples
        filt[n] += Yi[c, n]
    end

    envelope!(raw, FS)
    envelope!(filt, FS)
    peak = max(maximum(abs, raw), maximum(abs, filt))
    scale = 0.89 / peak                                  # marge anti-écrêtage

    out = vcat(raw .* scale, filt .* scale)
    pcm = Vector{Int16}(undef, length(out))
    @inbounds for i in eachindex(out)
        pcm[i] = round(Int16, clamp(out[i], -1, 1) * 32_767)
    end

    # Vérification intégrée. Une première tentative jugeait l'effet avec un passe-haut à un
    # pôle : il laissait passer les fondamentales (110–392 Hz), que le filtre ne touche pas,
    # et concluait à tort que la démo était inaudible. On mesure donc l'énergie DANS une bande
    # précise, par Goertzel, là où le passe-bas doit vraiment mordre.
    function goertzel(x, f, fs)
        ω = 2π * f / fs
        c = 2cos(ω)
        s1 = s2 = 0.0
        @inbounds for v in x
            s0 = v + c * s1 - s2
            s2 = s1; s1 = s0
        end
        sqrt(max(s1^2 + s2^2 - c * s1 * s2, 0)) / length(x)
    end
    println()
    println("vérification — amplitude par bande, brut → filtré")
    for f in (200.0, 600.0, 1500.0, 3000.0)
        a = goertzel(raw, f, FS)
        b = goertzel(filt, f, FS)
        println(@sprintf("  %6.0f Hz : %8.4f → %8.4f   (%5.1f dB)",
                         f, a, b, 20log10(max(b, 1e-12) / max(a, 1e-12))))
    end
    println()

    path = joinpath(@__DIR__, "biquad_demo.pcm")
    write(path, pcm)
    println(@sprintf("écrit %s — %d échantillons, %.1f s", path, length(pcm), length(pcm)/FS))
    println("puis :  ffmpeg -y -f s16le -ar $FS -ac 1 -i $path -b:a 96k <sortie>.mp3")
end

abspath(PROGRAM_FILE) == (@__FILE__) && main()

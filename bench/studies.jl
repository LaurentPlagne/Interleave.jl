# Les études comparatives, par opposition aux bancs par noyau de `runall.jl`.
#
# `runall.jl` répond à « que vaut le DLI sur ce noyau ». Les fichiers ci-dessous répondent à
# « que vaudrait l'alternative » : une bibliothèque du domaine, un SoA écrit à la main, et
# l'effet de la complexité de la récurrence.
#
#   julia --project=bench bench/studies.jl        # tour rapide, tailles modestes
#
# ⚠️ Ce fichier passe des tailles RÉDUITES pour tenir en quelques minutes et en mémoire
# raisonnable. Les tableaux publiés dans `docs/src/manual/what-it-replaces.md` viennent des
# **valeurs par défaut** de chaque banc, qui sont plus grosses — jusqu'à ~5 Go pour
# `pentasoa.jl`. Pour les reproduire, lancer le fichier voulu directement :
#
#   julia --project=bench bench/dspcompare.jl     # Interleave contre DSP.jl
#   julia --project=bench bench/soavsdli.jl       # tridiagonal : DLI contre SoA global
#   julia --project=bench bench/pentasoa.jl       # pentadiagonal : idem, scratch doublé
#   julia --project=bench bench/orderscan.jl      # IIR d'ordre M : balayage des prédécesseurs
#   julia --project=bench bench/videorec.jl       # vidéo : le verdict bascule avec M
#
# Chaque banc vérifie l'accord numérique AVANT de chronométrer, et exige la bit-exactitude du
# DLI contre la référence scalaire (AGENTS.md, invariant 1).

module SDsp;    include(joinpath(@__DIR__, "dspcompare.jl")); end
module SSoa;    include(joinpath(@__DIR__, "soavsdli.jl"));   end
module SPenta;  include(joinpath(@__DIR__, "pentasoa.jl"));   end
module SOrder;  include(joinpath(@__DIR__, "orderscan.jl"));  end
module SVideo;  include(joinpath(@__DIR__, "videorec.jl"));   end

SDsp.run(; nb = 256, ns = 2_048)
SSoa.run(; batches = (1_024, 16_384, 262_144))
SPenta.run(; batches = (16_384, 65_536))
SOrder.run(; nb = 8_192, ns = 512, orders = (1, 2, 4, 8))
SVideo.run(; nb = 64, orders = (0, 1, 2, 4))
println()

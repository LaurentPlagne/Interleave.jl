# Lance tous les bancs. Usage :  julia --project=bench -t auto bench/runall.jl
#
# ⚠️ Chaque banc tourne dans son propre module : les fichiers définissent tous des
# `run`, `reference!` et `batched` homonymes.
module BThomas;   include(joinpath(@__DIR__, "thomas.jl"));        end
module BBiquad;   include(joinpath(@__DIR__, "biquad.jl"));        end
module BDepth;    include(joinpath(@__DIR__, "depthwise.jl"));     end
module BVideo;    include(joinpath(@__DIR__, "video.jl"));         end
module BOption;   include(joinpath(@__DIR__, "optionpricing.jl")); end
module BReduce;   include(joinpath(@__DIR__, "reductions.jl"));    end

BThomas.run()
BBiquad.run()
BDepth.run()
BVideo.run()
BOption.run()
BReduce.run()
println()

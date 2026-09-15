using Test
using Interleave
using KernelAbstractions
using InteractiveUtils: code_llvm

include("kernels.jl")

# Tampon de travail préalloué et partagé : un type nommé, pas une closure (une closure
# capturant une variable englobante est boxée et alloue à chaque appel du driver).
struct SharedBuf{S}
    buf::S
end
(s::SharedBuf)() = s.buf

# Keep allocation measurements behind a function barrier.  Measuring a keyword call
# directly in the top-level testset can count the test harness' boxed globals on stable
# Julia releases, even though the specialized driver itself is allocation-free.
function allocated_thomas(X, D, U, L, B, scratch)
    apply!(thomas!, X, D, U, L, B; scratch = scratch)
    @allocated apply!(thomas!, X, D, U, L, B; scratch = scratch)
end

function allocated_tridiag(R, D, U, L, X)
    apply!(tridiag_mul!, R, D, U, L, X)
    @allocated apply!(tridiag_mul!, R, D, U, L, X)
end

const T = Float32
const P = 8

# Chaque instance du lot doit être DIFFÉRENTE des autres : avec des instances
# identiques, un mélange de lanes (le bug le plus probable du layout) serait
# indétectable. Tous les jeux de données ci-dessous dépendent de l'indice de lot.
function thomas_setup(nbatch, nx; pack = Val(P))
    X, D, U, L, B = (Interleave.Array{T}(undef, nbatch, nx; pack) for _ in 1:5)
    fill!(X, 0); fill!(D, 2); fill!(U, -1); fill!(L, -1); fill!(B, 0)
    lb = B
    for b in 1:nbatch, i in 1:nx
        lb[b, i] = sinpi(T(b) / 8) + T(i) / nx
    end
    X, D, U, L, B
end

@testset "Interleave.jl" begin

    @testset "invariants de construction" begin
        @test_throws ArgumentError Interleave.Array{T}(undef, 16, 4; pack = Val(3))   # P non puissance de 2
        @test_throws ArgumentError Interleave.Array{T}(undef, 16, 4; pack = Val(0))
        @test_throws ArgumentError Interleave.Array{T}(undef, -1, 4; pack = Val(4))
        @test_throws ArgumentError Interleave.Array{T}(undef, 16; pack = Val(4))      # instance sans dimension
        A = Interleave.Array{T}(undef, 10, 4; pack = Val(4))
        @test packsize(A) == 4
        @test npacks(A) == 3            # cld(10,4)
        @test npadding(A) == 2
        @test size(A) == (10, 4)
        @test instance_size(A) == (4,)
        @test_throws DimensionMismatch Interleave.Array(parent(A), 99)
    end

    @testset "layout DLI et vue scalaire" begin
        A = Interleave.Array{T}(undef, 8, 3; pack = Val(4))
        fill!(A, 0)
        la = A
        for b in 1:8, i in 1:3
            la[b, i] = 10b + i
        end
        @test all(la[b, i] == 10b + i for b in 1:8, i in 1:3)       # aller-retour
        # Le layout attendu : lane p contiguë, puis i, puis paquet.
        flat = reinterpret(reshape, T, parent(A))                   # (P, nx, npacks)
        @test flat[3, 2, 1] == la[3, 2]      # lot 3 = lane 3 du paquet 1
        @test flat[1, 2, 2] == la[5, 2]      # lot 5 = lane 1 du paquet 2
        # Padding : défini (zéro), invisible depuis la vue scalaire.
        Apad = Interleave.Array{T}(undef, 6, 2; pack = Val(4))
        flatpad = reinterpret(reshape, T, parent(Apad))
        @test all(iszero, flatpad[3:4, :, 2])
        @test size(Apad, 1) == 6

        # L'invariant vaut aussi en enveloppant un stockage existant et via `similar`.
        raw = fill(Vec{4,T}(1), 2, 2)
        wrapped = Interleave.Array(raw, 6)
        flatwrapped = reinterpret(reshape, T, parent(wrapped))
        @test all(iszero, flatwrapped[3:4, :, 2])

        S = similar(Apad)
        flatsimilar = reinterpret(reshape, T, parent(S))
        @test all(iszero, flatsimilar[3:4, :, 2])
    end

    @testset "construction depuis une compréhension" begin
        source = T[10b + i / 10 for b in 1:10, i in 1:4]
        A = @inferred Interleave.Array{T,2,8}(source)
        B = @inferred Interleave.Array(source; pack = Val(4))
        C = Interleave.Array{Float64,2,8}(source)

        @test A isa Interleave.Array{T,2,8}
        @test B isa Interleave.Array{T,2,4}
        @test C isa Interleave.Array{Float64,2,8}
        @test A == B == source
        @test C == Float64.(source)
        @test all(iszero, A.flat[3:8, :, 2])       # padding initialisé après la copie
    end

    @testset "similar préserve le layout lorsque la forme le permet" begin
        A = Interleave.Array{T}(undef, 10, 4; pack = Val(4))
        S = @inferred similar(A)
        S64 = @inferred similar(A, Float64)
        Saxes = @inferred similar(A, Float64, axes(A))
        strings = @inferred similar(A, String)
        resized = @inferred similar(A, T, (6, 3, 2))
        vector = @inferred similar(A, T, (7,))

        @test S isa Interleave.Array{T,2,4}
        @test size(S) == size(A)
        @test S64 isa Interleave.Array{Float64,2,4}
        @test size(S64) == size(A)
        @test Saxes isa Interleave.Array{Float64,2,4}
        @test axes(Saxes) == axes(A)
        # Un type que `SIMD.Vec` ne représente pas se replie sur le stockage standard.
        @test strings isa Matrix{String}
        @test resized isa Interleave.Array{T,3,4}
        @test size(resized) == (6, 3, 2)
        # Une dimension ne peut pas porter à la fois le lot et une instance : repli Base.
        @test vector isa Vector{T}
        @test size(vector) == (7,)
    end

    @testset "Thomas — invariant inverse (résidu)" begin
        nbatch, nx = 61, 32                       # 61 non multiple de 8 : padding actif
        X, D, U, L, B = thomas_setup(nbatch, nx)
        apply!(thomas!, X, D, U, L, B; scratch = similar(instance(X, 1)))
        R = Interleave.Array{T}(undef, nbatch, nx; pack = Val(P))
        fill!(R, 0)
        apply!(tridiag_mul!, R, D, U, L, X)    # on remultiplie : T*x doit redonner b
        lr, lb = R, B
        residu = maximum(abs(lr[b, i] - lb[b, i]) for b in 1:nbatch, i in 1:nx)
        @test residu < 1f-4                       # tolérance : Float32, n=32, matrice bien conditionnée
    end

    @testset "bit-exactitude scalaire ↔ vectorisé" begin
        # L'invariant central : la lane p du résultat vectoriel doit être EXACTEMENT
        # (==, pas ≈) le résultat scalaire de l'instance correspondante. C'est ce test
        # qui valide le layout, le driver et le padding d'un seul coup.
        nbatch, nx = 61, 32
        X, D, U, L, B = thomas_setup(nbatch, nx)
        apply!(thomas!, X, D, U, L, B; scratch = similar(instance(X, 1)))
        lx, lb = X, B
        for b in 1:nbatch
            xs = zeros(T, nx); ds = fill(T(2), nx); us = fill(T(-1), nx)
            ls = fill(T(-1), nx); bs = T[lb[b, i] for i in 1:nx]; ss = zeros(T, nx)
            thomas!(xs, ds, us, ls, bs, ss)       # même noyau, éléments scalaires
            @test all(lx[b, i] === xs[i] for i in 1:nx)
        end
    end

    @testset "biquad — gain continu et bit-exactitude" begin
        nbatch, nsamp = 61, 256
        b0, b1, b2, a1, a2 = T.((0.2f0, 0.4f0, 0.2f0, -0.3f0, 0.1f0))
        coeffs = (b0, b1, b2, a1, a2)
        Xin = Interleave.Array{T}(undef, nbatch, nsamp; pack = Val(P))
        Yout = Interleave.Array{T}(undef, nbatch, nsamp; pack = Val(P))
        fill!(Yout, 0); fill!(Xin, 0)
        lx = Xin
        amp(b) = T(1 + b / 100)                   # amplitude propre à chaque instance
        for b in 1:nbatch, n in 1:nsamp
            lx[b, n] = amp(b)
        end
        apply!((Y, X) -> biquad!(Y, X, coeffs), Yout, Xin)
        ly = Yout
        # Régime établi sur une entrée constante : y → H(1)*x avec H(1)=Σb/(1+Σa).
        gain = (b0 + b1 + b2) / (1 + a1 + a2)
        @test all(isapprox(ly[b, nsamp], gain * amp(b); rtol = 1f-4) for b in 1:nbatch)
        # Bit-exactitude contre le même noyau en scalaire
        for b in (1, 7, 8, 9, 61)
            ys = zeros(T, nsamp); xs = fill(amp(b), nsamp)
            biquad!(ys, xs, coeffs)
            @test all(ly[b, n] === ys[n] for n in 1:nsamp)
        end
    end

    @testset "P quelconque, résultats identiques" begin
        nbatch, nx = 61, 32
        ref = nothing
        for pk in (Val(1), Val(2), Val(4), Val(8), Val(16), Val(32))
            X, D, U, L, B = thomas_setup(nbatch, nx; pack = pk)
            apply!(thomas!, X, D, U, L, B; scratch = similar(instance(X, 1)))
            got = [X[b, i] for b in 1:nbatch, i in 1:nx]
            ref === nothing ? (ref = got) : @test(got == ref)   # égalité exacte
        end
    end

    @testset "stabilité de type" begin
        X, D, U, L, B = thomas_setup(16, 8)
        @test @inferred(instance(X, 1)) isa SubArray
        @test @inferred(X[3, 4]) isa T
        @test @inferred(packsize(X)) == P
        @test @inferred(apply!(thomas!, X, D, U, L, B; scratch = similar(instance(X, 1)))) isa Interleave.Array
    end

    @testset "le driver n'alloue pas" begin
        # Un tampon partagé n'est légitime qu'en séquentiel ; il sert ici à isoler les
        # allocations propres au driver de celles du tampon.
        nx = 32
        shared = SharedBuf(fill!(Array{Vec{P,T}}(undef, nx), zero(Vec{P,T})))
        for nbatch in (64, 6400)                   # l'invariant : aucune croissance avec npacks
            X, D, U, L, B = thomas_setup(nbatch, nx)
            @test allocated_thomas(X, D, U, L, B, shared) == 0
            R, Dm, Um, Lm, Xm = thomas_setup(nbatch, nx)
            @test allocated_tridiag(R, Dm, Um, Lm, Xm) == 0
        end
    end

    @testset "parallèle == séquentiel, et déterministe" begin
        # Le seul test qui distingue une course d'une erreur de calcul : égalité au
        # séquentiel ET reproductibilité sur plusieurs exécutions (guide §9).
        nbatch, nx = 1001, 32
        Xs, D, U, L, B = thomas_setup(nbatch, nx)
        apply!(thomas!, Xs, D, U, L, B; scratch = similar(instance(Xs, 1)))
        seq = [Xs[b, i] for b in 1:nbatch, i in 1:nx]
        for sched in (StaticScheduler(), DynamicScheduler(), StaticScheduler(chunking = false))
            for _ in 1:3
                Xp, Dp, Up, Lp, Bp = thomas_setup(nbatch, nx)
                parallel_apply!(thomas!, Xp, Dp, Up, Lp, Bp;
                          scratch = similar(instance(Xp, 1)), scheduler = sched)
                @test [Xp[b, i] for b in 1:nbatch, i in 1:nx] == seq
            end
        end
    end

    @testset "packtype : P=1 est scalaire" begin
        # `Vec{1,T}` empêcherait LLVM de vectoriser la boucle contiguë qu'il sait faire
        # seul : à P=1 le conteneur doit redevenir un tableau dense ordinaire.
        @test packtype(Float32, Val(1)) === Float32
        @test packtype(Float32, Val(8)) === Vec{8,Float32}
        A1 = Interleave.Array{T}(undef, 10, 4; pack = Val(1))
        A8 = Interleave.Array{T}(undef, 10, 4; pack = Val(8))
        # L'objet expose toujours des scalaires ; c'est la représentation machine qui change.
        @test eltype(A1) === eltype(A8) === T
        @test eltype(parent(A1)) === T
        @test eltype(parent(A8)) === Vec{8,T}
        @test packtype(A1) === T && packtype(A8) === Vec{8,T}
        @test parent(A1) isa Matrix{T}
        @test packsize(A1) == 1 && npacks(A1) == 10
        @test instance(A1, 3) isa SubArray{T,1}
    end

    @testset "un lot est un tableau de scalaires" begin
        # Le contrat d'usage : on manipule le lot sans jamais mentionner P.
        A = Interleave.Array{T}(undef, 10, 4; pack = Val(8))
        fill!(A, 0)
        @test A isa AbstractArray{T,2}
        @test size(A) == (10, 4)
        @test eltype(A) === T
        A[3, 2] = 1.5
        @test A[3, 2] === 1.5f0                       # un scalaire, pas un paquet
        A .= reshape(1:40, 10, 4)                     # broadcast depuis un tableau
        @test A[3, 2] == 13
        A .+= 1
        @test A[3, 2] == 14
        @test sum(A) == sum(reshape(1:40, 10, 4)) + 40
        M = fill(T(7), 10, 4)
        copyto!(A, M)
        @test A == M                                  # comparable à un Array ordinaire
        @test Array(A) isa Matrix{T}
        @test collect(A) == M
        @test A[:, 1] == fill(T(7), 10)
        @test size(@view A[2:4, :]) == (3, 4)
        @test map(x -> 2x, A) == 2M
        # Indexation linéaire : doit parcourir la vue LOGIQUE, pas l'entrelacement.
        A .= reshape(1:40, 10, 4)
        @test [A[i] for i in eachindex(A)] == collect(reshape(1:40, 10, 4))
        @test all(A[i] === reshape(Float32.(1:40), 10, 4)[i] for i in eachindex(A))
        @test A[17] === A[7, 2]
        A[23] = -5
        @test A[3, 3] === -5.0f0

        # Instances 3-D : indexation scalaire à trois indices logiques.
        B = Interleave.Array{T}(undef, 6, 3, 4; pack = Val(4))
        fill!(B, 0)
        B[5, 2, 3] = 7
        @test size(B) == (6, 3, 4)
        @test B[5, 2, 3] === 7.0f0

        # Le même code utilisateur, quel que soit P, donne le même résultat.
        function remplir!(C)
            for b in axes(C, 1), i in axes(C, 2)
                C[b, i] = b + i / 10
            end
            C
        end
        energie(C) = sum(abs2, C)
        vals = [energie(remplir!(Interleave.Array{T}(undef, 10, 4; pack = Val(P))))
                for P in (1, 2, 4, 8, 16, 32)]
        @test all(==(first(vals)), vals)              # le rembourrage ne fuit pas
    end

    @testset "lanetype" begin
        @test lanetype(Vec{8,Float32}) === Float32
        @test lanetype(Float32) === Float32
        @test lanetype(Vec{4,Float64}(1)) === Float64
        # L'usage qui le motive : oftype/convert échouent sur Vec.
        @test_throws MethodError oftype(Vec{4,Float32}(1), 0.125)
        @test Vec{4,Float32}(1) * lanetype(Vec{4,Float32})(0.125) === Vec{4,Float32}(0.125f0)
    end

    @testset "instances 2D — depthwise et Sobel" begin
        nchan, H, W = 61, 12, 10
        w = T.((1, 2, 1, 2, 4, 2, 1, 2, 1) ./ 16)      # noyau normalisé : somme = 1
        O = Interleave.Array{T}(undef, nchan, H, W; pack = Val(P))
        I = Interleave.Array{T}(undef, nchan, H, W; pack = Val(P))
        fill!(O, 0); fill!(I, 0)
        @test instance_size(O) == (H, W)
        @test size(O) == (nchan, H, W)
        li = I
        for c in 1:nchan, i in 1:H, j in 1:W
            li[c, i, j] = c + i / 10 + j / 100      # chaque canal diffère
        end
        apply!((o, i) -> depthwise3x3!(o, i, w), O, I)
        lo = O
        # Bit-exactitude contre le même noyau en scalaire
        for c in (1, 8, 9, 61)
            os = zeros(T, H, W)
            is = T[li[c, i, j] for i in 1:H, j in 1:W]
            depthwise3x3!(os, is, w)
            @test all(lo[c, i, j] === os[i, j] for i in 2:H-1, j in 2:W-1)
        end
        # Invariant : un noyau de somme 1 appliqué à un champ constant rend ce constant
        Ic = Interleave.Array{T}(undef, nchan, H, W; pack = Val(P))
        Oc = Interleave.Array{T}(undef, nchan, H, W; pack = Val(P))
        fill!(Ic, 3); fill!(Oc, 0)
        apply!((o, i) -> depthwise3x3!(o, i, w), Oc, Ic)
        @test all(isapprox(Oc[c, i, j], 3; rtol = 1f-6)
                  for c in 1:nchan, i in 2:H-1, j in 2:W-1)

        # Sobel : sur un champ spatialement constant les deux gradients s'annulent,
        # il ne reste que le terme temporel β*(curr-prev)².
        Os = Interleave.Array{T}(undef, nchan, H, W; pack = Val(P))
        Cs = Interleave.Array{T}(undef, nchan, H, W; pack = Val(P))
        Ps = Interleave.Array{T}(undef, nchan, H, W; pack = Val(P))
        fill!(Os, 0); fill!(Cs, 2); fill!(Ps, 0.5)
        α, β = 0.7f0, 0.3f0
        apply!((o, c, p) -> sobel_motion!(o, c, p, (α, β)), Os, Cs, Ps)
        attendu = β * (2f0 - 0.5f0)^2
        @test all(isapprox(Os[c, i, j], attendu; rtol = 1f-5)
                  for c in 1:nchan, i in 2:H-1, j in 2:W-1)
    end

    @testset "Black-Scholes Crank-Nicolson" begin
        nopt, ngrid, nt = 61, 24, 8
        V, D, U, L, R = (Interleave.Array{T}(undef, nopt, ngrid; pack = Val(P)) for _ in 1:5)
        fill!(D, 2.05); fill!(U, -0.5); fill!(L, -0.5); fill!(R, 0); fill!(V, 0)
        lv = V
        for b in 1:nopt, i in 1:ngrid
            lv[b, i] = max(T(i) - T(b) / 10, 0)      # payoff propre à chaque option
        end
        payoff = [lv[b, i] for b in 1:nopt, i in 1:ngrid]
        apply!((v, d, u, l, r, s) -> blackscholes_cn!(v, d, u, l, r, s, nt),
                  V, D, U, L, R; scratch = similar(instance(V, 1)))
        @test all(isfinite, V)
        # Bit-exactitude contre le même noyau en scalaire
        for b in (1, 8, 9, 61)
            vs = payoff[b, :]
            ds = fill(T(2.05), ngrid); us = fill(T(-0.5), ngrid); ls = fill(T(-0.5), ngrid)
            rs = zeros(T, ngrid); ss = zeros(T, ngrid)
            blackscholes_cn!(vs, ds, us, ls, rs, ss, nt)
            @test all(V[b, i] === vs[i] for i in 1:ngrid)
        end
    end

    @testset "vue scalaire : même mémoire" begin
        A = Interleave.Array{T}(undef, 10, 4; pack = Val(4))
        fill!(A, 0)
        @test A.flat isa Array{T,3}                 # un vrai Array, pas un ReinterpretArray
        @test size(A.flat) == (4, 4, 3)             # (P, nx, npacks)
        # Les trois vues partagent le buffer : une écriture est vue par les deux autres.
        A.flat[2, 3, 1] = 7
        @test A[2, 3] == 7
        @test parent(A)[3, 1][2] == 7
        A[5, 1] = -1                          # lot 5 = lane 1 du paquet 2
        @test A.flat[1, 1, 2] == -1
        @test parent(A)[1, 2][1] == -1
    end

    @testset "instances 3-D et au-delà" begin
        # Le conteneur ne privilégie aucune dimension : une instance peut être un volume.
        A = Interleave.Array{T}(undef, 20, 4, 5, 6; pack = Val(8))
        @test instance_size(A) == (4, 5, 6)
        @test size(A) == (20, 4, 5, 6)
        @test size(instance(A, 1)) == (4, 5, 6)
        B4 = Interleave.Array{T}(undef, 20, 3, 4, 5, 6; pack = Val(4))
        @test instance_size(B4) == (3, 4, 5, 6)
        @test size(B4) == (20, 3, 4, 5, 6)

        # Noyau 3-D écrit en indices naturels, comparé au même noyau en scalaire.
        nb, n1, n2, n3 = 17, 6, 7, 8
        O = Interleave.Array{T}(undef, nb, n1, n2, n3; pack = Val(8))
        I = Interleave.Array{T}(undef, nb, n1, n2, n3; pack = Val(8))
        fill!(O, 0); fill!(I, 0)
        li = I
        for b in 1:nb, i in 1:n1, j in 1:n2, k in 1:n3
            li[b, i, j, k] = b + i / 10 + j / 100 + k / 1000
        end
        apply!(laplacien3d!, O, I)
        for b in (1, 9, 17)
            is = T[li[b, i, j, k] for i in 1:n1, j in 1:n2, k in 1:n3]
            os = zeros(T, n1, n2, n3)
            laplacien3d!(os, is)
            @test all(O[b, i, j, k] === os[i, j, k]
                      for i in 2:n1-1, j in 2:n2-1, k in 2:n3-1)
        end
    end

    @testset "choix de l'axe paqueté" begin
        # Une même grille (nx, ny, nz), paquetée sur y puis sur z : c'est l'ordre des
        # arguments du constructeur qui décide, le noyau ne change pas.
        nx, ny, nz = 12, 16, 8
        Py = Interleave.Array{T}(undef, ny, nx, nz; pack = Val(4))
        Pz = Interleave.Array{T}(undef, nz, nx, ny; pack = Val(4))
        @test instance_size(Py) == (nx, nz)
        @test instance_size(Pz) == (nx, ny)
        @test packsize(Py) == packsize(Pz) == 4

        # Balayage de Thomas le long de x, vectorisé à travers y.
        X, D, U, L, B = (Interleave.Array{T}(undef, ny, nx, nz; pack = Val(4)) for _ in 1:5)
        fill!(X, 0); fill!(D, 2); fill!(U, -1); fill!(L, -1); fill!(B, 0)
        lb = B
        for y in 1:ny, i in 1:nx, k in 1:nz
            lb[y, i, k] = sinpi(y / 4) + i / nx + k / nz
        end
        apply!(thomas_lines!, X, D, U, L, B;
                  scratch = () -> Vector{packtype(X)}(undef, nx))
        for y in (1, 5, 16)
            for k in 1:nz
                xs = zeros(T, nx, 1)
                ds = fill(T(2), nx, 1); us = fill(T(-1), nx, 1); ls = fill(T(-1), nx, 1)
                bs = T[lb[y, i, k] for i in 1:nx, _ in 1:1]
                ss = zeros(T, nx)
                thomas_lines!(xs, ds, us, ls, bs, ss)
                @test all(X[y, i, k] === xs[i, 1] for i in 1:nx)
            end
        end
    end

    @testset "un Base.Array standard est un lot valide" begin
        # Le principe de Legolas++ : on écrit et on teste l'algorithme avec un tableau
        # ordinaire, puis on ne change que le type pour gagner en vitesse.
        nb, nx = 23, 16
        std(v) = fill(T(v), nb, nx)
        Xs, Ds, Us, Ls, Bs = std(0), std(2), std(-1), std(-1), std(1)
        @test packsize(Xs) == 1
        @test npacks(Xs) == nb
        @test instance_size(Xs) == (nx,)
        @test npadding(Xs) == 0
        @test packtype(Xs) === T
        @test size(instance(Xs, 3)) == (nx,)
        apply!(thomas!, Xs, Ds, Us, Ls, Bs; scratch = similar(instance(Xs, 1)))

        leg(v) = (A = Interleave.Array{T}(undef, nb, nx; pack = Val(8)); fill!(A, v); A)
        Xl, Dl, Ul, Ll, Bl = leg(0), leg(2), leg(-1), leg(-1), leg(1)
        apply!(thomas!, Xl, Dl, Ul, Ll, Bl; scratch = similar(instance(Xl, 1)))

        # Même noyau, même driver, deux dispositions mémoire : résultats bit-identiques.
        @test all(Xs[b, i] === Xl[b, i] for b in 1:nb, i in 1:nx)

        # Instances 2-D sur un tableau standard.
        Os, Is = zeros(T, 9, 6, 5), fill(T(1), 9, 6, 5)
        @test size(instance(Is, 2)) == (6, 5)
        w = T.((1, 2, 1, 2, 4, 2, 1, 2, 1) ./ 16)
        apply!((o, i) -> depthwise3x3!(o, i, w), Os, Is)
        @test all(≈(1), Os[b, i, j] for b in 1:9, i in 2:5, j in 2:4)

        # Mélanger des dispositions incompatibles est refusé.
        @test_throws DimensionMismatch apply!(thomas!, Xs, Dl, Us, Ls, Bs;
                                                 scratch = similar(instance(Xs, 1)))
    end

    @testset "driver GPU — contrat batch-major sur backend CPU" begin
        # Le backend CPU de KernelAbstractions teste exactement le wrapper et le noyau
        # de lancement employés sur Metal, sans rendre la suite dépendante d'un GPU.
        nb, nx = 23, 16
        std(v) = fill(T(v), nb, nx)
        X, D, U, L, B, S = std(0), std(2), std(-1), std(-1), std(1), std(0)
        ref = copy(X)
        apply!(thomas!, ref, D, U, L, B; scratch = zeros(T, nx))

        @test @inferred(gpu_backend(X)) isa KernelAbstractions.CPU
        @test @inferred(gpu_apply!(thomas!, X, D, U, L, B;
                                   scratch = S, workgroupsize = 8, wait = true)) === X
        @test X == ref
        @test gpu_synchronize(X) === nothing

        # Une instance 2-D conserve ses indices naturels dans le noyau utilisateur.
        O, I = zeros(T, nb, 6, 5), fill(T(1), nb, 6, 5)
        w = T.((1, 2, 1, 2, 4, 2, 1, 2, 1) ./ 16)
        gpu_apply!((o, i) -> depthwise3x3!(o, i, w), O, I;
                   workgroupsize = 8, wait = true)
        @test all(==(1), O[b, i, j] for b in 1:nb, i in 2:5, j in 2:4)

        @test_throws ArgumentError gpu_apply!(thomas!)
        @test_throws ArgumentError gpu_apply!(thomas!, X, D; workgroupsize = 0)
        @test_throws DimensionMismatch gpu_apply!(thomas!, X, D[1:end-1, :])
        @test_throws DimensionMismatch gpu_apply!(thomas!, X, D; scratch = S[1:end-1, :])
        @test_throws ArgumentError gpu_apply!(identity, zeros(T, nb))
        @test gpu_apply!(identity, zeros(T, 0, nx); wait = true) isa Matrix{T}
    end

    @testset "driver GPU — tous les noyaux de validation" begin
        # Le backend CPU de KernelAbstractions exécute le même chemin de lancement que
        # Metal/CUDA. Chaque entrée est une fonction nommée, afin que le compilateur GPU
        # puisse l'inliner sans dépendre d'une closure capturante.
        nb, ns = 7, 19
        X = fill(T(1), nb, ns); Y = zeros(T, nb, ns)
        ref = copy(Y); apply!(gpu_biquad!, ref, X)
        gpu_apply!(gpu_biquad!, Y, X; wait = true)
        @test Y == ref

        H, W = 9, 8
        I = fill(T(1), nb, H, W); O = zeros(T, nb, H, W)
        ref = copy(O); apply!(gpu_depthwise3x3!, ref, I)
        gpu_apply!(gpu_depthwise3x3!, O, I; wait = true)
        @test O == ref

        C = fill(T(2), nb, H, W); P = fill(T(0.5), nb, H, W)
        O .= 0; ref .= 0
        apply!(gpu_sobel_motion!, ref, C, P)
        gpu_apply!(gpu_sobel_motion!, O, C, P; wait = true)
        @test O == ref

        ngrid = 13
        V = [max(T(i) - T(b) / 10, 0) for b in 1:nb, i in 1:ngrid]
        D = fill(T(2.05), nb, ngrid); U = fill(T(-0.5), nb, ngrid)
        L = fill(T(-0.5), nb, ngrid); R = zeros(T, nb, ngrid)
        S = zeros(T, nb, ngrid); Vref = copy(V); Rref = copy(R); Sref = copy(S)
        apply!(gpu_blackscholes_cn!, Vref, D, U, L, Rref; scratch = Sref)
        gpu_apply!(gpu_blackscholes_cn!, V, D, U, L, R; scratch = S, wait = true)
        @test V == Vref

        n1, n2, n3 = 6, 7, 8
        I3 = [T(b) + T(i) / 10 + T(j) / 100 + T(k) / 1000
              for b in 1:nb, i in 1:n1, j in 1:n2, k in 1:n3]
        O3 = zeros(T, nb, n1, n2, n3); R3 = copy(O3)
        apply!(laplacien3d!, R3, I3)
        gpu_apply!(laplacien3d!, O3, I3; wait = true)
        @test O3 == R3

        nx, m = 11, 3
        XL = zeros(T, nb, nx, m); DL = fill(T(2), nb, nx, m)
        UL = fill(T(-1), nb, nx, m); LL = fill(T(-1), nb, nx, m)
        BL = [sinpi(T(b) / 8) + T(i + c) / nx for b in 1:nb, i in 1:nx, c in 1:m]
        SL = zeros(T, nb, nx); XLref = copy(XL); SLref = copy(SL)
        # A single device work item sees a (nx,m) instance and a (nx) scratch row.
        for b in 1:nb
            thomas_lines!(view(XLref, b, :, :), view(DL, b, :, :), view(UL, b, :, :),
                          view(LL, b, :, :), view(BL, b, :, :), view(SLref, b, :))
        end
        gpu_apply!(thomas_lines!, XL, DL, UL, LL, BL; scratch = SL, wait = true)
        @test XL == XLref

        R = zeros(T, nb, ngrid); Xr = fill(T(0.25), nb, ngrid)
        ref = copy(R); apply!(tridiag_mul!, ref, D, U, L, Xr)
        gpu_apply!(tridiag_mul!, R, D, U, L, Xr; wait = true)
        @test R == ref
    end

    @testset "apply! ne parallélise pas" begin
        # Contrat : le parallélisme doit être visible au point d'appel. `apply!` n'a
        # aucun mot-clé qui l'autoriserait — c'est vérifié structurellement, pas par
        # convention.
        kw(f) = Base.kwarg_decl(first(methods(f)))
        @test :scratch in kw(apply!)
        @test :scheduler ∉ kw(apply!)
        @test :nchunks ∉ kw(apply!)
        @test :scheduler in kw(parallel_apply!)     # et parallel_apply!, lui, l'expose
        @test :nchunks in kw(parallel_apply!)
    end

    @testset "packs et Base.foreach" begin
        nb, nx = 61, 32
        X, D, U, L, B = thomas_setup(nb, nx)
        pX = packs(X)
        @test pX isa AbstractVector
        @test length(pX) == npacks(X)
        # `==` entre vues de Vec rendrait un Vec{8,Bool} : on compare les vues elles-mêmes.
        @test pX[2] === instance(X, 2)

        # `foreach` a exactement la bonne sémantique et rend `nothing`.
        buf = Vector{packtype(X)}(undef, nx)
        r = foreach((x, d, u, l, b) -> thomas!(x, d, u, l, b, buf),
                    pX, packs(D), packs(U), packs(L), packs(B))
        @test r === nothing

        # …et le même résultat que le driver, au bit près.
        Y, Dy, Uy, Ly, By = thomas_setup(nb, nx)
        apply!(thomas!, Y, Dy, Uy, Ly, By; scratch = similar(instance(Y, 1)))
        @test all(X[b, i] === Y[b, i] for b in 1:nb, i in 1:nx)

        # …sans allocation (la spécialisation de foreach évite le zip générique de Base).
        gofe() = foreach((x, d, u, l, b) -> thomas!(x, d, u, l, b, buf),
                         pX, packs(D), packs(U), packs(L), packs(B))
        gofe()
        @test @allocated(gofe()) == 0

        # `packs` marche aussi sur un Base.Array.
        S = fill(T(1), 7, 5)
        @test length(packs(S)) == 7
        @test size(packs(S)[3]) == (5,)

        # Un lot vide est valide et son itérateur reste entièrement inféré.
        E = Interleave.Array{T}(undef, 0, 5; pack = Val(P))
        pE = @inferred packs(E)
        @test isempty(pE)
        @test eltype(pE) === typeof(instance(X, 1))
        called = Ref(false)
        @test foreach(_ -> (called[] = true), pE) === nothing
        @test !called[]
        @test apply!(_ -> (called[] = true), E) === E
        @test parallel_apply!(_ -> (called[] = true), E) === E
        @test !called[]

        ES = Matrix{T}(undef, 0, 5)
        pES = @inferred packs(ES)
        @test isempty(pES)
        @test eltype(pES) === typeof(instance(S, 1))

        @test_throws DimensionMismatch foreach((a, b) -> nothing, packs(X), packs(S))
    end

    @testset "substitution de type par alias" begin
        # Le contrat central : un Interleave.Array se construit EXACTEMENT comme un
        # Base.Array, `P` étant un paramètre de type et non un mot-clé — sans quoi on ne
        # pourrait pas le loger dans un alias.
        function resoudre(Arr, nsys, nx)
            X, D, U, L, B = (Arr(undef, nsys, nx) for _ in 1:5)
            fill!(X, 0); fill!(D, 2); fill!(U, -1); fill!(L, -1); fill!(B, 1)
            apply!(thomas!, X, D, U, L, B; scratch = similar(instance(X, 1)))
            X
        end
        ref = resoudre(Base.Array{T}, 61, 32)
        @test ref isa Matrix{T}
        for P in (1, 2, 4, 8, 16, 32)
            got = resoudre(Interleave.Array{T,2,P}, 61, 32)
            @test packsize(got) == P
            @test size(got) == size(ref)
            @test all(got[b, i] === ref[b, i] for b in 1:61, i in 1:32)
        end
        # instances 2-D, même substitution
        @test size(Interleave.Array{T,3,8}(undef, 7, 3, 5)) == size(Base.Array{T}(undef, 7, 3, 5))
        # le 2e paramètre est bien le nombre de dimensions, comme chez Base
        @test ndims(Interleave.Array{T,3,8}(undef, 7, 3, 5)) == ndims(Base.Array{T,3}(undef, 7, 3, 5)) == 3
        @test_throws ArgumentError Interleave.Array{T,3,8}(undef, 7, 3)
        # la forme à mot-clé reste disponible quand P est calculé
        @test packsize(Interleave.Array{T}(undef, 10, 4; pack = Val(4))) == 4
    end

    @testset "corroboration structurelle : LLVM vectorisé" begin
        # Un speedup ne prouve pas la vectorisation (cache, ILP) : on exige le
        # <P x float> dans le LLVM émis (AGENTS.md §3).
        for pk in (4, 8, 16, 32)
            io = IOBuffer()
            code_llvm(io, thomas!, NTuple{6,Vector{Vec{pk,T}}}; debuginfo = :none)
            @test occursin("<$pk x float>", String(take!(io)))
        end
        io = IOBuffer()
        code_llvm(io, biquad!, Tuple{Vector{Vec{8,T}},Vector{Vec{8,T}},NTuple{5,T}};
                  debuginfo = :none)
        @test occursin("<8 x float>", String(take!(io)))
    end
end

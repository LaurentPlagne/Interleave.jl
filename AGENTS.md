# AGENTS.md — Interleave.jl

Règles de travail pour ce dépôt. Les règles **générales** de style, de performance et
d'outillage Julia sont dans [`julia-recommandations.md`](julia-recommandations.md) :
le présent fichier ne les répète pas, il **renvoie** aux sections pertinentes et ajoute
ce qui est **propre à ce projet**.

## 1. De quoi il s'agit

Portage en Julia des concepts de **Legolas++** (C++, `../Legolas`) : vectoriser des
algorithmes **intrinsèquement séquentiels** (récurrences : Thomas tridiagonal, IIR
biquad, schémas temporels) en les appliquant à un **lot de problèmes indépendants**
entrelacés en mémoire — *Data Layout Interleaving* (DLI).

L'idée tient en une phrase : **le même code source scalaire devient vectoriel par le
seul changement du type d'élément** (`Float32` → `Vec{P,Float32}`). Tout le reste
(layout, driver, threads) est de la plomberie autour de cette phrase.

Conséquence de conception : ce paquet ne doit **pas** porter l'implémentation C++
(shapes récursives, traits par niveau, allocateur, expression templates). `AbstractArray`,
les vues et `reinterpret` couvrent déjà ce besoin. Porter l'**idée**, pas le contournement
des limites de C++.

## 2. Les invariants du projet

0. **Un `Base.Array` standard est un lot valide.** `apply!` l'accepte, sa première
   dimension étant le lot et sa pack size valant 1. C'est ce qui permet d'écrire et de
   tester un algorithme avant de choisir sa disposition mémoire — le principe même de
   Legolas++, où `Legolas::Array<T,D>` est déjà un tableau rectangulaire N-D et où le
   packing n'est qu'un jeu de paramètres supplémentaires. Toute évolution du driver doit
   préserver ce chemin, et l'égalité bit à bit entre les deux dispositions.

0 bis. **`apply!` ne parallélise pas, et ne doit jamais le pouvoir.** Seul `parallel_apply!`
   lance des tâches. Une bibliothèque qui threade en silence casse tout appelant déjà
   placé dans une région parallèle ; le parallélisme doit se voir au point d'appel. C'est
   la raison d'être du `parmap` explicite de Legolas++. Concrètement : **ne jamais ajouter
   de mot-clé `scheduler`, `nchunks` ou équivalent à `apply!`**, ni à la méthode
   `Base.foreach` sur `Packs`. Un test vérifie l'absence de ces mots-clés.

0 ter. **Un lot est un tableau de scalaires.** `size(A) == (nbatch, dims…)`, `eltype(A) === T`,
   `A[b, i, j]` rend un scalaire, et l'indexation **linéaire** parcourt cette vue logique.
   Le layout n'est accessible que par `parent(A)`. Toute méthode ajoutée à `Interleave.Array`
   doit respecter ce contrat.
   - ⚠️ Contraindre `getindex`/`setindex!` à `Vararg{Int,N}` avec `N = ndims` : sans cela
     `A[17]` est capté comme un numéro d'instance et rend silencieusement le mauvais
     élément. Bug réellement rencontré, désormais testé.

Ce sont les propriétés à contrôler **à la construction ou dans les tests**
(`julia-recommandations.md` §6), parce qu'une régression silencieuse y est indétectable
à l'œil.

1. **Bit-exactitude scalaire ↔ vectorisé.** Pour tout noyau, la lane `p` du résultat
   `Vec{P,T}` doit être **exactement** égale (`==`, pas `≈`) au résultat scalaire de
   l'instance correspondante. C'est le test central de la bibliothèque : il valide en
   même temps le layout, le driver et le padding.
   - Corollaire (§5.2) : **jamais `@fastmath` dans un noyau ni dans le driver.** La
     contraction FMA change l'ordre des opérations et casserait l'égalité. Si un FMA est
     voulu, l'écrire explicitement en `muladd` — des deux côtés.
   - Corollaire : ne pas réordonner une somme ni remplacer `inv(s)` par une division
     (§5.1) sous prétexte d'optimisation.
2. **`P` est une puissance de deux.** `sizeof(Vec{P,T}) == P*sizeof(T)` n'est garanti que
   dans ce cas (LLVM peut padder `Vec{3,Float32}`), et tout le `reinterpret` en dépend.
   À vérifier dans le constructeur.
3. **Le padding ne fuit pas.** `npack*P ≥ nbatch` : les lanes au-delà de `nbatch`
   calculent des valeurs sans signification. Elles ne doivent apparaître dans aucune
   réduction ni aucune sortie. Les initialiser (pas de `undef` propagé en `NaN`/`Inf`,
   qui peut ralentir certains matériels et pollue le débogage).
4. **Le chemin chaud ne touche pas de `ReinterpretArray`.** Le stockage natif est un
   `Array{Vec{P,T},N}` dense, accessible par `parent(A)` ; l'indexation scalaire de l'objet
   la dérive et est réservée au remplissage, aux comparaisons et aux I/O. L'inverse
   (stocker scalaire, voir packé) est le choix du C++ ; il est plus lent ici.
5. **Le résultat parallèle est déterministe et égal au séquentiel.** C'est le seul test
   qui distingue une course d'une erreur de calcul (§9). À lancer plusieurs fois.

### Invariants GPU expérimentaux

6. **Un work item GPU possède une instance complète.** La récurrence reste séquentielle
   dans ce work item ; le groupe SIMT parallélise le lot. Ne pas transférer `Vec{P,T}` sur
   le GPU et ne pas écrire une boucle explicite sur des lanes dans le noyau.
7. **Le stockage device est scalaire et batch-major** : `(nbatch, dims…)`. La première
   dimension contiguë rend coalescés les accès de work items voisins au même indice de
   récurrence. `P` n'existe pas sur ce chemin ; le réglage est `workgroupsize`.
8. **Aucun transfert implicite.** `gpu_apply!` reçoit des tableaux déjà présents sur le
   device, soumet de façon asynchrone par défaut et ne copie rien. Le scratch GPU est lui
   aussi un lot device, une tranche privée par work item.
9. **Deux niveaux de maturité doivent rester explicites.** Metal passe par
   KernelAbstractions/Metal.jl et réutilise le noyau Julia. Vulkan consomme pour l'instant
   un shader GLSL compilé en SPIR-V avec un ABI figé ; ne pas prétendre à une source Julia
   unique tant qu'aucun compilateur Julia→SPIR-V maintenu ne la fournit.

## 2 bis. Quand le DLI paie — et quand il coûte

Mesuré par `bench/runall.jl` (M-series, NEON 128 bits, 8 threads, Float32). Le meilleur
`P` séquentiel, puis le même avec threads :

| Cas | récurrence ? | débit de la référence | meilleur DLI | + threads |
|---|:---:|---:|---:|---:|
| Thomas tridiagonal | oui (balayage) | 1.1 GFlop/s | P=16 : **13.4×** | 31.8× |
| IIR biquad | oui (temporelle) | 3.0 GFlop/s | P=16 : **15.0×** | 32.9× |
| Black-Scholes CN | oui (double) | 1.9 GFlop/s | P=16 : **11.8×** | **84.6×** |
| Depthwise 3×3 | **non** | 40.6 GFlop/s | P=16 : **0.43×** | 1.73× |
| Vidéo Sobel + mouvement | **non** | 43.4 GFlop/s | P=8 : **0.88×** | 2.77× |

**La règle de décision est le débit de la référence, pas le domaine applicatif.** Une
référence à 1–3 GFlop/s signale un compilateur en échec (récurrence) : le DLI rend 12 à
15×. Une référence à 40+ GFlop/s signale un compilateur qui a déjà vectorisé la boucle
interne : le DLI ne peut alors que **reprendre** ce gain, jamais l'ajouter — et il le
reprend mal.

Mécanisme à retenir : **`Vec{1,T}` bloque la vectorisation que LLVM aurait faite seul.**
Sur les noyaux sans récurrence, `P=1` est 4 à 10× *plus lent* que la référence, parce que
l'élément-vecteur d'une lane empêche l'auto-vectorisation le long de l'axe contigu. Sur
les noyaux à récurrence, `P=1` colle à la référence à 2 % près — il n'y avait rien à
bloquer.

Conséquence : pour un stencil ou une convolution sans dépendance, garder le **même
noyau** et choisir `P = 1` (§2 ter), ce qui rend la parité avec la référence. Les colonnes
`P ≥ 2` du tableau ci-dessus mesurent le mauvais réglage pour ces deux cas, pas une limite
de la bibliothèque.

Le README C++ annonce 5.8× sur la convolution depthwise ; c'est un gain contre une
référence *scalaire*, pas contre une référence vectorisée par le compilateur. Ne pas
reproduire cette comparaison ici.

**Le threading et le SIMD ne se multiplient que si le noyau est compute-bound.**
Black-Scholes (11.8× × 7.2 ≈ 85×) réutilise sa grille en cache sur `nt` pas de temps ;
Thomas (13.4× → 31.8×, soit 2.4× seulement sur 8 threads) est limité par la bande
passante. Vérifier de quel régime on parle avant d'annoncer un gain composé.

## 2 ter. Le type d'élément EST le réglage — jamais un second noyau

**Le principe du projet : on écrit le noyau une fois, et on choisit le type pour obtenir
la meilleure performance sur une cible donnée.** Tout ce qui oblige l'utilisateur à écrire
une seconde version du noyau (une boucle de lane explicite, une variante « à plat ») viole
ce principe, quel que soit son gain. Une telle variante a été écrite puis **retirée** : elle
gagnait 4 % sur un stencil au prix d'un noyau dupliqué.

Le réglage est `P`, et il porte jusqu'au type d'élément :

```julia
packtype(T, Val(1)) === T             # tableau dense ordinaire
packtype(T, Val(P)) === Vec{P,T}      # paquet vectoriel
```

⚠️ **`P == 1` doit rendre `T`, jamais `Vec{1,T}`.** Un vecteur à une lane empêche LLVM de
vectoriser la boucle contiguë qu'il aurait vectorisée seul : mesuré 4.0 GFlop/s contre
40.2 sur le même stencil, soit **10× de pénalité**. Avec le type scalaire, `P = 1` redevient
exactement le tableau de référence — et le même noyau couvre les deux régimes.

Mesuré (`bench/`), même noyau à chaque ligne, seul `P` change :

| noyau | meilleur `P` | débit | contre la référence |
|---|:---:|---:|---:|
| Thomas (récurrence) | 16 | 15.4 GFlop/s | 13.4× |
| IIR biquad (récurrence) | 16 | 44.4 GFlop/s | 15.0× |
| Black-Scholes (double récurrence) | 16 | 22.0 GFlop/s | 11.8× |
| Depthwise 3×3 (sans récurrence) | **1** | 40.2 GFlop/s | 1.05× |
| Vidéo Sobel (sans récurrence) | **1** | — | ≈1× |

La règle de choix reste le débit de la référence (§2 bis) : un noyau que le compilateur
vectorise déjà veut `P = 1`, un noyau à récurrence veut `P` grand.

### Détail d'implémentation : `A.flat`

`A.flat` est la vue scalaire `(P, dims…, npacks)` du **même buffer**, construite par
`unsafe_wrap` dans le constructeur. Elle sert à implémenter l'indexation scalaire de
`Interleave.Array` ; ce n'est pas une seconde façon d'écrire un noyau.

Elle **doit** être un vrai `Array` et jamais un `ReinterpretArray` : mesuré sur un même
stencil, `unsafe_wrap` → 46.8 GFlop/s, `reinterpret(reshape, T, data)` → 12.4, soit
**3.8×** pour des octets identiques. Sûreté : `flat` n'est pas propriétaire, mais `data` —
champ de la même structure immuable — maintient la mémoire en vie ; aucun `GC.@preserve`
n'est requis chez l'appelant, et un `A.flat` ne doit pas fuir hors de la vie de son `A`.

### Expositions écartées, avec leurs chiffres

Encapsuler le paquet « à la volée » au-dessus d'un stockage scalaire (la conception du C++,
`getPackedView()`) a été mesuré sous quatre formes sur Thomas P=16, contre 2.17 ms pour le
stockage natif : pointeur brut 2.79 ms (×1.29), `reinterpret` 3.44 ms (×1.58),
`vload(Vec, vecteur, i)` 4.92 ms (×2.27), boucle `p` écrite à la main 9.77 ms (×4.50).
Hisser le `reinterpret` hors de la boucle ne change rien — le coût est dans l'**indexation**,
pas dans la construction de la vue. L'alignement non plus (`vloada` ≡ `vload`).

## 3. Performance : comment mesurer ici

Voir §3 du guide. Ce qui s'applique particulièrement :

- **`@allocated` est la métrique robuste.** Un noyau appelé par pack doit être à **zéro
  allocation** : le scratch est alloué **une fois par tâche** par le driver, jamais dans
  le noyau (§3.7 — le piège du tampon réalloué). Un test d'allocation vaut mieux qu'un
  chronomètre.
- **Chronométrage en A/B entrelacé** (§3.3) : alterner les variantes et prendre le
  minimum, relever `uptime` avant. Un chiffre isolé sur machine chargée ne prouve rien.
- **Corroboration structurelle** obligatoire pour toute affirmation de vectorisation :
  `@code_llvm debuginfo=:none` doit contenir `<P x float>`. Un speedup seul ne prouve pas
  que la vectorisation a eu lieu (il peut venir du cache ou de l'ILP) — et
  réciproquement, un `<P x float>` présent ne garantit pas le gain. Les deux, ou rien.
- **`P` n'est pas la largeur SIMD du matériel.** Mesuré sur ce Mac (NEON 128 bits, donc
  4×Float32), Thomas 65 536×64, `bench/thomas.jl` :

  | P | 1 | 2 | 4 | 8 | 16 |
  |---|---|---|---|---|----|
  | speedup | 1.00× | 2.0× | 4.0× | 7.2× | **13.4×** |

  `P=16` vaut quatre fois la largeur matérielle et reste le meilleur : une récurrence est
  *latency-bound*, et plusieurs vecteurs en vol cachent la latence de la chaîne de
  dépendance. **Donc `P` se règle par mesure, par noyau** — ne jamais le coder en dur
  « à la largeur du CPU ». `P=1` reproduit la référence naïve exactement — à `P = 1` le
  conteneur *est* un tableau dense ordinaire (§2 ter).
- **Le threading rapporte peu une fois `P` bien choisi** : `P=16` passe de 13.4× à 16.2×
  sur 8 threads. À ce régime le noyau est limité par la bande passante mémoire (5 tableaux
  parcourus), pas par le calcul. Ne pas vendre les deux gains comme multiplicatifs.
- **Threads** : éviter `Threads.@threads`, dont la latence de lancement est élevée devant
  la taille d'un chunk ici. Utiliser `OhMyThreads` (schedulers explicites).
- ⚠️ **Mesurer derrière une barrière de fonction.** Une globale non-`const` dans le
  harnais de mesure (§2.1), ou une variable de boucle portant une fonction (dispatch
  dynamique), fausse `@allocated` de plusieurs centaines d'octets et envoie sur de
  fausses pistes. Chaque mesure dans sa propre fonction, tout en local.

## 3 bis. Signatures : jamais de type abstrait

**Ni `f::Function`, ni `args::Tuple` nu.** Soit l'argument n'est pas typé, soit il porte
un **type paramètre** : `f::F where F`, `args::NTuple{NA,Any} where NA`,
`arrays::Vararg{`Interleave.Array`,NA} where NA`.

Une annotation abstraite empêche la spécialisation : l'appel devient dynamique et les
arguments sont boxés. **Mesuré sur le driver** : avec `arrays::Tuple` et `views::Tuple`,
240 octets alloués *par paquet* (192 Ko pour 800 paquets) ; après paramétrage, 144 octets
*constants*, et **zéro** sur le chemin sans scratch.

Corollaire, mesuré aussi : une **closure** capturant une variable englobante est boxée
(144 octets par appel du driver). Lui préférer un **type nommé appelable** — c'est
pourquoi `scratchlike` rend un `ScratchProto` et non une closure.

## 4. Tests

Voir §7. Spécifique au projet :

- Chaque noyau d'exemple (Thomas, Biquad) sert de **test de non-régression numérique** :
  oracle scalaire naïf sur `Matrix` d'un côté, `apply!` de l'autre, comparaison exacte.
- `@inferred` sur le driver et sur l'accès aux vues — une instabilité de type y annule
  tout le bénéfice (et n'apparaît pas dans les tests de justesse). ⚠️ `@inferred` n'est
  pas compté dans le résumé de `@testset` (§7.3).
- Tester `nbatch` **non multiple de `P`** systématiquement : c'est le cas qui casse.
- Exécuter la suite **dans la session chaude** (`include("test/runtests.jl")` via `ex`),
  pas en sous-processus (§7.1, §10.7).

## 5. Outillage

- **Tout le Julia passe par `ex`** (Kaimon), jamais par `julia` en ligne de commande :
  le REPL est partagé avec l'utilisateur et l'état persiste (§10.1).
- **Paquets** : `pkg_add` / `pkg_rm`, jamais `Pkg.add` ni `Pkg.activate` (§10.4).
- `Manifest.toml` **n'est pas versionné** en phase de développement (§8).
- Recherche de code : `search_code` pour un concept, `grep_code` pour un token exact,
  `search_methods` pour les méthodes d'une générique (§10.5). Le C++ de référence est
  **hors projet** (`../Legolas`) : là seulement le shell est légitime.

## 6. Points ouverts

- ~~Le nom `Legolas` est déjà pris dans le registre General.~~ **Réglé** : le paquet
  s'appelle `Interleave` (`Project.toml`), le dépôt est `Interleave.jl`, et le nom
  `Interleave` est libre dans le registre General (vérifié le 2026-09-16). Seul le
  répertoire de travail local s'appelle encore `Legolas.jl`.
- **Toute reconstruction champ par champ casse l'alias `data`/`flat`.** `deepcopy`
  (`src/array.jl`) et `Serialization` (`ext/InterleaveSerializationExt.jl`) sont traités ;
  JLD2, BSON et Arrow ne le sont pas et reproduiraient le bug — silencieusement, puisque
  l'indexation scalaire écrirait alors dans un tampon qu'aucun noyau ne lit.
- Composition `Vec{P,Dual}` (SIMD × différentiation automatique) : ne fonctionne pas
  directement (`Vec` exige un type feuille LLVM). La généricité vaut sur chaque axe
  séparément, pas encore sur leur produit.

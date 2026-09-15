# Recommandations Julia — guide indépendant du projet

Extrait générique de `AGENTS.md` et des pièges **mesurés** en session. Rien ici ne
dépend d'un projet particulier : ces règles valent pour tout code Julia numérique.
Elles sont classées par thème ; les ⚠️ marquent ce qui a réellement mordu.

## 1. Dispatch et conception des types

1. **Dispatch multiple** : préférer le dispatch sur les types aux `if/else` ou à une
   approche orientée objet. Une variante de comportement = un type, une méthode.
2. **Réifier ce que le code hérité encode.** Un état plat indexé à la main
   (`y[6*(i-1)+k]`), un mode sélectionné par un entier global, un algorithme choisi par
   un drapeau : tout cela devient des types nommés et du dispatch.
3. **Le nom du champ EST le nom de l'accesseur.** N'écrire aucune fonction qui ne fait
   qu'écho à un champ. Une fonction d'accès ne se justifie que si elle **abstrait** ou
   **assemble** (ex. `norme`, `densité`), ou si elle sert en broadcast.
4. **Ne pas homogénéiser une asymétrie** : deux grandeurs de natures différentes gardent
   deux champs. Un tuple homogène masque l'information au lieu de la nommer.
5. **Envelopper un `NTuple` dans un type nommé, jamais un alias** : surcharger une
   opération sur un alias (`const Vec3 = NTuple{3,Float64}`) changerait le comportement
   de *tout* tuple de cette forme dans le module. Un `struct Vec3; data::NTuple{3,Float64}; end`
   avec ses méthodes `+`, `-`, `*`, `cross`, `squared_norm` est le bon geste.
6. **Paramétrer plutôt que dupliquer** : une variante (potentiel, dimension, précision)
   est un paramètre de type ou un type de dispatch, pas une copie de chaque boucle.
   `struct Forces{E<:PairPotential,I<:PairPotential}` plutôt que des drapeaux.
7. **Pas de splat inutile** : `f(map(g, t)...)` qui reconstruit aussitôt un tuple est un
   aller-retour. Le splat ne sert qu'à **concaténer**.
8. Préférer le **vocabulaire de `Base`** à un nom inventé : `reverse(t)` plutôt qu'un
   `swap_xxx` maison.
9. Pour un état scalaire mutable capturé par une closure, préférer `Ref` ou un `mutable
   struct` dédié à un `Vector` d'un élément.

## 2. Stabilité des types

1. **Aucun global non `const`** dans les chemins chauds : les valeurs voyagent dans un
   contexte passé en argument.
2. **Type de retour prévisible** : éviter les `Union` inutiles et `Any`. Une fonction
   qui retourne tantôt un `Float64`, tantôt `nothing` se découpe.
3. **`@inferred` dans les tests** pour chaque brique publique, pas seulement les
   accesseurs : y compris les fonctions récursives.
4. `@code_warntype` (via `InteractiveUtils`) pour localiser une instabilité.
5. **Tuples de taille fixe** : `ntuple(f, Val(N))` est inféré ; `ntuple(f, n::Int)`
   runtime ne l'est pas. Préférer `Val`.

## 3. Performance et mémoire

1. Convention **`!`** pour les mutations ; **`@views`** pour le slicing (aucune copie) ;
   **vectorisation par l'opérateur point** ; **`Tuple`/`NTuple`** dès que la taille est
   fixe (pile, immuable, zéro allocation tas).
2. **`eachcol`/`eachrow`** pour itérer sur des colonnes sans copie.
3. **Mesurer, jamais supposer** :
   - `@allocated` est la métrique **robuste** (indépendante de la charge machine) ;
     un compteur d'allocations, un nombre de passes mémoire, un nombre d'opérations
     tiennent sous n'importe quelle charge.
   - Pour le temps : relever la charge (`uptime`) AVANT de chronométrer, et préférer
     l'**A/B entrelacé** (alterner deux variantes, prendre le minimum) qui annule les
     dérives lentes.
   - Chercher une **corroboration structurelle** : un chiffre seul sur machine chargée ne
     prouve rien.
4. **Profilage CPU** : `using Profile; Profile.clear(); @profile f();`
   puis `io = IOBuffer(); Profile.print(io, Profile.fetch(); format=:flat, mincount=…)`.
   ⚠️ L'IO de `Profile.print` est **positionnel** : `Profile.print(io, data; …)` ;
   avec le mot-clé `io=`, la sortie est vide.
5. **Profilage des allocations** :
   `Profile.Allocs.clear(); Profile.Allocs.@profile sample_rate=1.0 f();`
   puis agréger `Profile.Allocs.fetch().allocs` par **première frame de votre module**
   (chaque `Alloc` a `.size` et `.stacktrace`). C'est ce qui a permis de trouver que
   93 % des allocations venaient d'un unique constructeur de tampons.
6. ⚠️ **Profiler sans job concurrent** : un REPL partagé qui exécute un calcul en
   parallèle contamine le profil.
7. ⚠️ **Ne pas réallouer dans une récursion** : un `particle_set(n)` (ou tout tampon)
   neuf à chaque appel récursif peut dominer le coût. Parade : un *workspace* aux
   tampons préalloués, un jeu par niveau de récursion si le parcours est en profondeur
   (un seul appel vit par niveau), réutilisé sur toute la boucle externe. Mesuré :
   ÷190 sur le total d'allocations, sans changer les résultats.
8. **Mutualiser aussi les masques** (`BitVector`) et les sorties (`devnull`) plutôt que
   de les recréer.
9. **Coopérativité** pour un calcul long : vérifier un drapeau d'annulation dans la
   boucle, exposer une progression, sauvegarder des checkpoints — un calcul long non
   coopératif n'est ni observable ni interruptible.

## 4. Style : le fonctionnel est un moyen, la lisibilité tranche

1. `map`, `filter`, `any`, `all`, `reduce`, `findfirst`, compréhensions et générateurs
   plutôt que des boucles à accumulateurs ou drapeaux. Fonctions courtes en affectation
   directe (`f(x) = …`) pour prédicats, accesseurs, transformations simples.
2. La compacité doit **préserver `@inferred` et l'absence d'allocations**.
3. ⚠️ **Deux formes à proscrire** :
   - un `foldl` dont on jette le résultat et qui mute des accumulateurs extérieurs
     (c'est une boucle déguisée) ;
   - `Iterators.peel(Iterators.filter(…))` sur un générateur imbriqué pour trouver le
     premier élément. Une boucle avec un drapeau **nommé** se lit mieux.
4. Pas de `let` dans une définition en forme d'affectation pour se donner une variable
   locale : c'est exactement ce que fournit un corps de `function … end`.
5. Pas d'**arithmétique d'indices déguisée** : `t[3-i]` pour « l'autre extrémité »
   devient `reverse(t)[i]`. Et `x / 2` plutôt que `0.5 * x` (lisibilité et arithmétique
   identique).
6. Ternaire pour un choix simple : `t <= t0 && return 0.0` ; `amp = t > t1 ? 1.0 : …`.
7. **Réifier un accumulateur** (état de courant, compteur d'énergie) dans un
   `mutable struct` avec un constructeur par défaut, plutôt que trois scalaires promenés.
8. **One-liners** : pour un prédicat, un accesseur dérivé ou une transformation simple,
   une ligne est idéale si elle porte **une seule idée** :
   `positive(x) = x > 0` ; `box_length(p) = (4π / 3 * n)^(1 / 3) * r_ws` ;
   `softened(r, ε) = r^2 + ε^2`. Au-delà d'une idée ou de ~92 colonnes, revenir à une
   fonction multiligne : un one-liner illisible est un dette, pas une élégance.
9. **Raccourcis et sucre syntaxique** (quand ils restent lisibles) :
   - sortie anticipée : `t <= t0 && return 0.0` ;
   - garde : `isfile(path) || return` ;
   - comparaisons chaînées : `0 <= i < n` ;
   - déstructuration : `x, y = y, x` ; retour structuré en `NamedTuple`
     (`(nu = …, sigma = …, n = …)`) plutôt qu'un tuple anonyme ;
   - itération : `for (i, r) in enumerate(eachcol(A))`, `zip(xs, ws)`, blocs `do`
     pour les callbacks.
   Un `&&` ou un `||` dont l'effet de bord est caché se remplace par un `if` nommé.
10. **Ternaires** pour choisir une **expression** : `dt = t > t1 ? 1.0 : (t - t0) / (t1 - t0)`.
    Pas de ternaire imbriqué sur plus de deux niveaux, pas de branche à effet de bord.
    Le ternaire et le court-circuit sont aussi des raccourcis **performants** : ils
    évitent des branches explicitement nommées et se compilent en sauts directs.
11. **Types paramétrés pour spécialiser** : `struct Box{T<:Real}; side::T; end`,
    `struct Forces{E<:Potential,I<:Potential}; …; end`. Le paramètre de type **spécialise
    la compilation** (performance) et remplace un drapeau d'exécution
    (`if kind == :a …`), donc code plus compact **et** plus rapide. `Val{N}` pour une
    taille connue à la compilation (`ntuple(f, Val(N))`, un tuple inféré). Règle : ne
    paramétrer que si le paramètre change une méthode ou une branche compilée.
12. **Broadcast fusionné** : `c = @. a + b * x` combine les opérations sans tableau
    intermédiaire ; `@views A[:, 1]` évite une copie. Compacité et performance vont ici
    de pair — mais vérifier avec `@allocated` que la fusion a bien eu lieu.

## 5. Exactitude numérique (quand on compare à une référence)

1. **L'ordre des opérations flottantes est un contrat.** Si le but est de reproduire
   une référence, ne pas réordonner les sommes, ne pas remplacer `1.0/(a*b)` par
   `(1.0/a)/b`, ni `x/l` par `x*(1.0/l)`.
2. Julia **ne contracte pas** en FMA sans `@fastmath`/`muladd` explicites : ne jamais
   activer `@fastmath` sur un noyau à comparer.
3. `x^2` se abaisse en `x*x` pour un littéral, mais `x^0.5`/`x^n` appelle `pow` :
   ne pas supposer une identité avec `sqrt`.
4. **Les fonctions spéciales ne sont pas toutes dans `Base`** : `erfc`, `erf`, `besselj`,
   `besselk`… Options sans dépendance : `ccall` vers OpenLibm
   (`erfc(x) = ccall(:erfc, Cdouble, (Cdouble,), x)`) ; sinon `SpecialFunctions.jl`
   (accord explicite requis si le projet limite ses dépendances).
5. ⚠️ **`Int(x)` lève `InexactError`** si `x` n'est pas entier — contrairement au cast C
   qui tronque. Écrire `trunc(Int, x)` (ou `floor`/`round` selon l'intention).
6. `sum` sur un **générateur** est un repli séquentiel ; sur un **tableau**, la somme
   est *pairwise*. Deux résultats différents possibles : choisir explicitement.
7. La libm de la plateforme peut différer de celle de la référence (1 ulp) : sur une
   trajectoire chaotique cela suffit à diverger après quelques dizaines de pas. En tirer
   la bonne conclusion : précision machine sur quelques pas, **barre de bruit** sur les
   observables longues.
8. Comparer deux configurations exige une **barre de bruit** : mesurer la dispersion sur
   plusieurs graines avant de conclure qu'une correction change quelque chose.

## 6. Vérifier les propriétés constatées, ne pas les supposer

- Une propriété *observée* (antisymétrie, conservation, cohérence de deux
  représentations) se contrôle **à la construction** ou dans les tests, pour qu'une
  régression s'arrête là plutôt que de se propager silencieusement.
- Contrôler les cas dégénérés (division par zéro, tableau vide) par des gardes dont le
  commentaire dit ce qu'elles **écartent**, pas seulement ce qu'elles testent.

## 7. Tests

1. `Test.jl` simple + `@testset`, suffisant dans la majorité des cas. Exécuter les tests
   **dans la session chaude** plutôt qu'en sous-processus (qui recompile tout).
2. ⚠️ La sortie d'un testset est un `println` comme un autre : récupérer l'objet rendu
   par `include` et le résumer (`Test.get_test_counts`), ou attraper la
   `Test.TestSetException` que `@testset` lève en cas d'échec.
3. ⚠️ `@inferred` **n'est pas compté comme un `@test`** dans le résumé : ne pas s'étonner
   d'un total inférieur au nombre de lignes.
4. Écrire des tests sur des **valeurs gelées** d'une référence (oracle), avec la
   tolérance justifiée dans un commentaire ; les préférer à des tests auto-référentiels.
5. ⚠️ Les filtres de `run_tests(pattern=…)` passent par `ARGS` et ne sont honorés que par
   les suites qui les lisent (ReTest) : sans cela, la suite complète tourne.
6. Vérifier les **invariants inverses** quand c'est possible : la fonction et son
   inverse, la somme et sa décomposition, la transformée et sa réciproque.

## 8. Outillage REPL

- **Environnements et dépendances** : en **phase de développement**, ne pas versionner
  `Manifest.toml` (l'ajouter au `.gitignore`) — il fige les versions exactes de la
  machine, provoque des conflits de fusion et brouille les diffs ; le `Project.toml`
  suffit. On ne le versionne que pour un environnement **reproductible** figé
  (application déployée, CI déterministe). Corollaire : ne pas éditer `Project.toml` à la
  main pour ajouter un paquet — passer par `Pkg.add` (ou l'outil du REPL), sinon le
  manifeste local se désynchronise.
- **Revise** recharge automatiquement : ne jamais appeler `Revise.revise()` (no-op).
  ⚠️ Revise suit les fichiers déjà inclus ; **l'ajout d'un nouvel `include` dans un
  module n'est pas toujours repris** (surtout si un calcul tourne en parallèle) :
  `Base.include(MonModule, "nouveau.jl")` une fois pour forcer.
- `InteractiveUtils.@code_warntype` pour la stabilité des types.
- JuliaFormatter (`format_code`) si installé dans l'environnement du projet.
- ⚠️ **Un eval long dans un REPL partagé** : le promouvoir en tâche de fond, ne pas
  sonder trop vite ; s'il bloque le REPL, seule issue : `kill -9` du processus puis
  redémarrer la session.

## 9. Pièges de bibliothèques (mesurés, génériques)

- **GLMakie/GLFW** : doit être chargé et utilisé sur **le fil principal** (thread 1),
  sinon `ThreadAssertionError` puis contexte OpenGL corrompu (« unalived context »),
  irrécupérable dans la session. Pas d'export **PDF** (PNG/JPEG seulement ; le PDF
  demande CairoMakie).
- **Makie** : `lines!` de certaines versions **n'accepte pas `marker`/`markersize`** →
  `scatterlines!`. `axislegend(ax, ["a","b"])` a été remplacé par des `label=` sur les
  tracés + `axislegend(ax; position=…)`. `ylims` n'est pas un attribut d'`Axis` →
  `ylims!(ax, lo, hi)`. `save` d'une Figure sur GLMakie ne fait que PNG/JPEG.
- **Sérialisation binaire maison** : `write(io, tuple)` n'existe pas ; écrire champ par
  champ (`write(io, Int64(x))`, `write(io, y)`), et relire dans le même ordre avec les
  mêmes types.
- ⚠️ **Closure et variable englobante homonymes** : un nom réutilisé entre une closure
  et sa fonction englobante devient **une seule variable boxée** — partagée par tous les
  fils. Vérifier qu'un résultat parallèle est **déterministe** sur plusieurs exécutions
  et **coïncide avec le séquentiel** : c'est le seul test qui distingue une course d'une
  erreur de calcul.
- ⚠️ **Une bibliothèque tierce n'est pas un oracle** : l'éprouver sur les données réelles
  et sur l'équation qu'elle prétend résoudre, jamais sur sa seule absence d'erreur.

## 10. Kaimon — bien l'utiliser

Kaimon est le REPL Julia partagé avec l'utilisateur (qui voit tout en direct) et le
portail d'outils d'intelligence de code. Le bon usage :

1. **Tout le Julia passe par `ex`**, jamais par `julia` en ligne de commande. Le REPL est
   partagé : le code évalué doit rester lisible, et l'état persiste entre les appels.
2. **Sortie** : `q=true` (défaut) pour ne pas rapatrier de valeur ; `q=false` uniquement
   quand on veut un résultat ciblé. `println`/`print` vers stdout sont **dépouillés** :
   terminer par une **expression** (souvent un `NamedTuple` ou un tuple) à afficher.
   Point-virgule systématique en fin de bloc pour supprimer l'écho.
3. **Sessions** : `investigate_environment()` pour l'état ; si aucun projet n'est
   connecté, `start_session(project_path=…)` le crée immédiatement (ne pas attendre
   l'utilisateur). Plusieurs sessions connectées → passer `ses=<clé>` à chaque outil.
   Seuls l'utilisateur peut autoriser un projet absent de la liste.
4. **Paquets** : `pkg_add(packages=[…])` / `pkg_rm`, jamais `Pkg.add` ni `Pkg.activate`
   (ne jamais changer de projet sous les pieds de la session).
5. **Les outils Kaimon d'abord, le shell ensuite** :
   - concept qu'on sait **décrire** → `search_code(query=…)` ;
   - **token exact** (symbole, appel, chaîne, TODO, regex) → `grep_code(pattern=…)`,
     `no_ignore=true` pour couvrir logs et généré ;
   - **toutes les méthodes d'une fonction générique** → `search_methods` (l'outil du
     dispatch multiple) ;
   - **champs, hiérarchie, sous-types** → `type_info` (sur un type **concret**) ;
   - symboles d'un fichier → `document_symbols` ; lire un fichier ou un log → `ex`.
   ⚠️ Les fichiers **hors du projet** (sources de référence, archives) échappent à ces
   outils : là seulement, le shell est légitime.
6. **« Qui appelle ceci ? »** n'a pas d'outil exact : `grep_code` est le filet de
   sécurité, et il **sur-déclare** (docstrings, commentaires) — c'est le bon sens de
   l'erreur. Une absence de référence n'est **jamais** une preuve de code mort (appels
   par valeur, fermetures).
7. **Tests** : les exécuter dans la **session chaude** (`include` du `runtests.jl` via
   `ex`), pas via `run_tests` (sous-processus qui recompile tout). ⚠️
   `run_tests(pattern=…)` ne filtre que les suites qui lisent `ARGS` (ReTest).
8. **`mt=true`** pour tout ce qui touche GLMakie/GLFW/GPU/affichage : ces codes exigent
   le fil principal. Le chargement de GLMakie **doit** être fait ainsi (sinon
   `ThreadAssertionError` puis contexte corrompu). `format_code` (JuliaFormatter) si
   disponible.
9. **Calculs longs** : au-delà de ~30 s, un `ex` **bascule en tâche de fond** et rend un
   `eval_id` ; le relever avec `check_eval` (attendre ≥30 s, puis ~60 s par appel) ;
   `cancel_eval` et `list_jobs` pour le contrôle. Écrire le calcul **coopératif** :
   `KaimonGate.is_cancelled()` dans la boucle, `KaimonGate.progress("…")`,
   `KaimonGate.stash(clé, valeur)` — ces noms sont `public` mais **non exportés** :
   les qualifier. ⚠️ `stash` attend une clé `String`.
10. ⚠️ **Un eval parti en boucle dans du code de bibliothèque bloque toute la session** :
    `cancel_eval` ne fait que lever un drapeau que ce code ne lit pas,
    `manage_repl(restart)` échoue sur une session bloquée, `1+1` reste en attente.
    Seule issue : `ping(extended=true)` pour le PID, `kill -9`, puis `start_session`.
    Corollaire : **tester les variantes une par une**, pas six dans un même eval.
11. ⚠️ **Les evals concurrents partagent l'état du REPL** (avertissement « shared REPL
    state may have changed concurrently ») : ne pas lancer de profil pendant qu'un job
    tourne, et se méfier des interactions entre evals.
12. **Processus externes** (oracles, scripts C…) : après une session tuée ou un timeout,
    vérifier les **zombies** (`ps`, `pgrep -fl`) et tuer par **PID** — un motif `pkill`
    ne matche pas toujours (`./ppbs` vs chemin complet).

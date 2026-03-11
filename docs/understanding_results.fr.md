# Comprendre les Résultats Open AI Box

## Vue d'ensemble

Quand vous exécutez `gr.export("graph.json")`, Open AI Box produit un fichier JSON
structuré avec quatre sections principales. Ce document explique chaque champ
et ce qu'il signifie concrètement.

---

## 1. `openaibox` — Métadonnées du package

```json
"openaibox": {
  "version": "1.0.0b1",
  "generated": "2025-03-11T12:00:00Z",
  "model": "HuggingFaceTB/SmolLM-360M"
}
```

| Champ | Signification |
|-------|--------------|
| `version` | Version de Open AI Box ayant produit l'analyse |
| `generated` | Timestamp UTC de l'analyse |
| `model` | Identifiant du modèle analysé |

---

## 2. `architecture` — Structure du modèle

```json
"architecture": {
  "class": "LlamaForCausalLM",
  "total_params": 361821120,
  "num_layers": 30,
  "hidden_dim": 960,
  "vocab_size": 49152
}
```

### `class`
Le nom de la classe Python du modèle. Indique la famille de transformeur :
- `LlamaForCausalLM` → Famille Llama / SmolLM / Mistral
- `Qwen2ForCausalLM` → Famille Qwen 2/2.5
- `GPT2LMHeadModel` → Famille GPT-2

### `total_params`
Nombre total de paramètres dans le modèle.
- 100M–500M → Modèle petit / mobile
- 1B–7B → Modèle consommateur standard
- 70B+ → Grand modèle frontier

### `num_layers`
Nombre de blocs transformer (couches décodeur). Chaque couche réalise :
self-attention + normalisation + projection feed-forward.

### `hidden_dim`
La taille du vecteur d'état caché principal.
C'est la dimension que Open AI Box cartographie dans `dimension_map`.
La représentation interne de chaque token est un vecteur de cette taille.

**Point clé** : La sortie du `decision_point` est `[1, 1, hidden_dim]`.
Les dimensions de hidden_dim portent l'intégralité du raisonnement du modèle.

### `vocab_size`
Nombre de tokens que le modèle connaît.
Le `lm_head` projette `hidden_dim → vocab_size` pour produire les probabilités de sortie.

---

## 3. `injection_points` — Où observer le modèle

C'est la section la plus exploitable.

```json
"injection_points": [
  {
    "name": "input_point",
    "role": "input",
    "layer": "model.embed_tokens",
    "in_shape": [1, 5],
    "out_shape": [1, 5, 960],
    "description": "Embeddings de tokens..."
  },
  {
    "name": "decision_point",
    "role": "decision",
    "layer": "model.norm",
    "in_shape": [1, 1, 960],
    "out_shape": [1, 1, 960],
    "description": "Normalisation finale..."
  }
]
```

### `role: "input"` — La couche d'embedding

Le prompt entre dans le modèle ici sous forme d'IDs de tokens `[1, seq_len]`
et est converti en vecteurs denses `[1, seq_len, hidden_dim]`.

**Lecture du shape** : `[1, 5, 960]` signifie :
- Taille de batch = 1
- Longueur de séquence = 5 tokens
- Dimension cachée = 960

### `role: "decision"` — ⭐ Le point le plus important

C'est la couche de normalisation finale — après que les 30 blocs transformer
ont traité l'entrée, mais avant que le modèle décide quoi dire ensuite.

Le shape de sortie `[1, 1, 960]` représente :
- Uniquement la dernière position de token (`seq=1` à l'inférence)
- Les 960 dimensions encodant la compréhension complète du modèle

**C'est là où le modèle "sait" tout ce qu'il sait, mais n'a pas encore parlé.**

### `role: "memory"` — Projections Key/Value

Les projections KV compriment le contexte pour le mécanisme d'attention.
Le ratio de shapes renseigne sur le Grouped Query Attention :
- `Q: [1, 5, 1536]` et `KV: [1, 5, 256]` → ratio 6:1 (style Qwen2.5)
- `Q: [1, 5, 960]` et `KV: [1, 5, 320]` → ratio 3:1 (style SmolLM)

Un ratio GQA plus élevé signifie une inférence plus efficace en mémoire.

### `role: "output"` — La tête du modèle de langage

`lm_head` est une projection linéaire : `hidden_dim → vocab_size`.
La sortie `[1, 1, 49152]` contient le score logit brut pour chaque token.

---

## 4. `layer_flow` — Trace d'exécution complète

```json
"layer_flow": [
  {"order": 0, "name": "model.embed_tokens", "type": "Embedding",
   "in": [[1, 5]], "out": [[1, 5, 960]], "params": 47185920},
  ...
]
```

Chaque entrée montre :
- `order` — Ordre d'exécution pendant l'inférence (0 = premier)
- `name` — Chemin du module PyTorch (correspond à `model.named_modules()`)
- `type` — Nom de la classe
- `in` / `out` — Shapes des tenseurs à cette couche
- `params` — Nombre de paramètres entraînables dans cette couche

**Lecture du flow :**
1. `embed_tokens` → convertit les IDs de tokens en vecteurs
2. `layers.0 → layers.N` → N blocs transformer traitent les vecteurs
3. `norm` → normalisation finale (le point de décision)
4. `lm_head` → projette vers le vocabulaire

---

## 5. `dimension_map` — Ce que chaque dimension porte

Présent uniquement si vous avez appelé `gr.map_dimensions()`.

```json
"dimension_map": {
  "hidden_dim": 960,
  "contrast_groups": {
    "syntax_semantics": {
      "question": "Quelles dimensions détectent l'absurdité sémantique ?",
      "cos_similarity": 0.6451,
      "separability": "VERY_SEPARABLE",
      "top_dimensions": [696, 792, 766, ...]
    }
  },
  "multi_role_dimensions": [
    {"index": 696, "roles": ["syntax_semantics", "causality", ...], "score": 0.91}
  ],
  "specialist_dimensions": {
    "causality": [295, 157],
    "certainty": [32, 545, 702, 683]
  }
}
```

### `cos_similarity` — À quel point les deux groupes sont-ils différents ?

| Valeur | Signification |
|--------|--------------|
| 0.60 | Très séparable — le modèle encode une différence claire |
| 0.85 | Séparable — différence détectable |
| 0.98 | Faible séparabilité — le modèle distingue à peine ces catégories |

**Découvertes clés pour SmolLM-360M :**
- `syntax_semantics` : cos=0.6451 → le modèle distingue clairement grammatical vs absurde
- `causality` : cos=0.9572 → le modèle distingue à peine cause et corrélation
- Cela explique pourquoi les petits LLMs confondent souvent "parce que" et "et"

### `multi_role_dimensions` — Régulateurs globaux

Les dimensions présentes dans le top-20 de 3+ groupes de contraste
sont des **régulateurs globaux**. Elles modulent l'intensité du signal
dans de nombreuses catégories simultanément.

Pour SmolLM : `dim_696` et `dim_544` sont dans les 6 catégories —
ce sont les dimensions individuelles les plus influentes du modèle.

### `specialist_dimensions` — Porteurs précis

Les dimensions présentes dans une seule catégorie sont des **spécialistes**.

Pour SmolLM :
- `[295, 157]` → portent l'information de **causalité** spécifiquement
- `[32, 545, 702, 683]` → portent l'information de **certitude** spécifiquement
- `[164, 93, 395]` → portent l'information **temporelle** spécifiquement

Ce sont les dimensions à cibler pour une analyse au niveau architectural.

---

## Lire un résultat en 60 secondes

1. Vérifiez `architecture.hidden_dim` → c'est la résolution du raisonnement du modèle
2. Trouvez dans `injection_points` le point avec `role == "decision"` → c'est la fenêtre d'observation principale
3. Consultez `dimension_map.contrast_groups` → trouvez les contrastes avec `cos_similarity` faible (< 0.90)
4. Regardez `specialist_dimensions` → ces dimensions portent des types spécifiques de raisonnement
5. Regardez `multi_role_dimensions` → les dims avec 5+ rôles sont les neurones les plus influents du modèle

---

## Questions fréquentes

**Q : Pourquoi le decision_point est-il si important ?**

R : C'est le seul endroit du modèle où toute l'information du contexte est intégrée
mais où aucune décision finale n'a encore été prise. C'est la "pensée complète"
du modèle avant la parole.

**Q : Que signifie une cos_similarity proche de 1.0 ?**

R : Le modèle traite les deux groupes de prompts presque identiquement.
Il ne distingue pas les concepts en question. Pour la causalité (0.9572),
cela signifie que SmolLM traite "parce que" et "et" de manière très similaire.

**Q : Puis-je analyser un modèle local (pas HuggingFace) ?**

R : Oui, si vous pouvez le charger avec PyTorch. Passez `model` et `tokenizer`
directement au constructeur de `Open AI Box`.

---

*Open AI Box v1.0.0-beta — [github.com/tryboy869/openaibox](https://github.com/tryboy869/openaibox)*

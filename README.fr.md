<div align="center">

<img src="assets/header_animation.svg" alt="Open AI Box — Open AI Box" width="900"/>

<br/>

<img src="assets/lock_openaibox.svg" alt="Open AI Box" width="520"/>

<br/><br/>

[![PyPI](https://img.shields.io/badge/pypi-v1.0.0--beta-58a6ff?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/openaibox/)
[![Licence: MIT](https://img.shields.io/badge/licence-MIT-3fb950?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-d2a8ff?style=flat-square&logo=python&logoColor=white)](https://python.org)

**Introspection universelle des LLM. N'importe quel modèle. N'importe quelle architecture.**

*Comprendre ce qui se passe à l'intérieur de n'importe quel modèle de langage — sans le modifier.*

[English](README.md) | **Français**

</div>

---

## Qu'est-ce que Open AI Box ?

Open AI Box est un package Python qui ouvre la boîte noire de n'importe quel LLM.

En traçant un passage d'inférence en direct avec les hooks forward de PyTorch,
Open AI Box découvre automatiquement :
- Le **graphe d'architecture complet** du modèle
- Les **points d'injection** — où les données entrent, où les décisions sont prises, où la mémoire réside
- Les **rôles des dimensions** — lesquelles du vecteur caché portent la causalité, l'émotion, la certitude, ou le raisonnement temporel

Tout est exporté dans un unique fichier `graph.json`.

---

## Installation

```bash
pip install openaibox
```

Dépendances : `torch`, `transformers`

---

## Démarrage rapide

```python
from openaibox import Open AI Box

# Charger et analyser n'importe quel modèle HuggingFace
gr = Open AI Box("HuggingFaceTB/SmolLM-360M")

# Étape 1 — Découvrir l'architecture
gr.discover()

# Étape 2 — Cartographier les rôles des dimensions (optionnel, analyse approfondie)
gr.map_dimensions()

# Étape 3 — Exporter vers graph.json
gr.export("graph.json")

# Étape 4 — Afficher le résumé
gr.print_summary()
```

---

## Concepts clés

### Points d'injection

Open AI Box identifie 4 types de points d'injection dans n'importe quel modèle :

| Rôle | Description |
|------|-------------|
| `input_point` | Où le prompt entre (embeddings de tokens) |
| `decision_point` | L'état de raisonnement le plus riche, juste avant la sélection du token |
| `memory_point` | Où les projections K/V encodent le contexte |
| `output_point` | Où les logits sur le vocabulaire sont calculés |

Le `decision_point` est le plus significatif : il contient la compréhension complète
du modèle sur le contexte avant de prendre une décision.

### Carte des dimensions

Chaque dimension du vecteur caché porte une information spécifique.
Open AI Box identifie :

- **Dimensions multi-rôles** — actives dans plusieurs catégories (régulateurs globaux)
- **Dimensions spécialistes** — actives dans une seule catégorie (porteurs précis)

### graph.json

```json
{
  "architecture": { "class": "LlamaForCausalLM", "hidden_dim": 960 },
  "injection_points": [
    { "role": "decision", "layer": "model.norm", "out_shape": [1, 1, 960] }
  ],
  "dimension_map": {
    "specialist_dimensions": {
      "causality": [295, 157],
      "certainty": [32, 545, 702, 683]
    }
  }
}
```

---

## Documentation

- [Comprendre les résultats (FR)](docs/understanding_results.fr.md)
- [Understanding Results (EN)](docs/understanding_results.md)
- [Changelog](CHANGELOG.md)

---

## Créateur

<div align="center">
<img src="assets/creator_card.svg" alt="Daouda Abdoul Anzize" width="680"/>
</div>

---

<div align="center">
<img src="assets/footer_animation.svg" alt="Open AI Box footer" width="900"/>
</div>

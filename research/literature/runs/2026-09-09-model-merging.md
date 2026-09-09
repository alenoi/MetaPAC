# Manual model-merging literature expansion — 2026-09-09

User-requested addition of the 13 papers identified in the preceding model-merging search. No scheduling changes.

Publications: 68 → 81 (+13). Search terms: 133 → 168 (+35). Duplicates among the 13 candidates: 0.

Deduplication: DOI/arXiv identifier and normalized title for publications; normalized terms across seed and discovered topics. All pre-existing entries and search rules retained.

Evidence: 13 additions screened from abstracts and primary metadata; full-text assessment remains pending. Conference venues recorded where verified; DELLA and ImPart retained as arXiv records without asserting an unverified conference venue. The preprint_date field denotes first arXiv submission and is distinct from the conference year. Terms are derived search phrases, not author-supplied keywords.

## Added publications

- [Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time](https://arxiv.org/abs/2203.05482) — `wortsman_model_soups_2022`. Averages models fine-tuned from a common initialization to improve accuracy and robustness. Useful weight-averaging baseline for studying compression before versus after merging in a MetaPAC extension.

- [Model Stock: All we need is just a few fine-tuned models](https://arxiv.org/abs/2403.19522) — `jang_model_stock_2024`. Uses weight-space geometry and layer-wise averaging to approximate a favorable center using two fine-tuned models. Relevant as a low-overhead merging baseline before adaptive compression.

- [Merging Models with Fisher-Weighted Averaging](https://arxiv.org/abs/2111.09832) — `matena_fisher_merging_2022`. Uses Fisher information to weight parameter contributions when merging compatible models. Provides a deterministic importance-based aggregation reference for a MetaPAC merging extension.

- [Dataless Knowledge Fusion by Merging Weights of Language Models](https://arxiv.org/abs/2212.09849) — `jin_regmean_2023`. RegMean merges language-model weights to reduce prediction differences relative to source models without requiring original training datasets. Relevant to reconstruction-guided aggregation and compression-aware model fusion.

- [Editing Models with Task Arithmetic](https://arxiv.org/abs/2212.04089) — `ilharco_task_arithmetic_2023`. Composes task vectors obtained by subtracting pretrained weights from fine-tuned weights. Establishes a baseline and a delta representation for extending MetaPAC importance estimation and compression to multiple fine-tuned models.

- [TIES-Merging: Resolving Interference When Merging Models](https://arxiv.org/abs/2306.01708) — `yadav_ties_merging_2023`. Trims small updates, resolves sign conflicts, and merges sign-consistent task-vector components. Relevant to importance-based retention and interference-aware aggregation of compressed updates.

- [Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch](https://arxiv.org/abs/2311.03099) — `yu_dare_2024`. DARE randomly drops and rescales delta parameters before applying a merging method. Provides a random-sparsification baseline against learned importance when compressing and merging fine-tuning updates.

- [AdaMerging: Adaptive Model Merging for Multi-Task Learning](https://arxiv.org/abs/2310.02575) — `yang_adamerging_2024`. Learns task-wise or layer-wise merging coefficients via entropy minimization on unlabeled test samples. Relevant as an adaptive allocation analogue, with its test-time data and optimization requirements distinguished from data-free merging.

- [DELLA-Merging: Reducing Interference in Model Merging through Magnitude-Based Sampling](https://arxiv.org/abs/2406.11617) — `deep_della_merging_2024`. Uses magnitude-ranked dropout probabilities and rescaling to sparsify updates before merging. Relevant importance-proxy baseline for a MetaPAC delta-compression and merging workflow.

- [Model Breadcrumbs: Scaling Multi-Task Model Merging with Sparse Masks](https://arxiv.org/abs/2312.06795) — `davari_model_breadcrumbs_2024`. Removes negligible and outlier task-vector updates before aggregation. Relevant to selective retention, sparse masks, and the distinction between update magnitude and downstream merging utility.

- [Task Singular Vectors: Reducing Task Interference in Model Merging](https://arxiv.org/abs/2412.00081) — `gargiulo_task_singular_vectors_2025`. Uses layer-wise SVD to compress task updates and reduce interference during merging. Closely related to extending MetaPAC with low-rank delta representations and compression-aware aggregation.

- [ImPart: Importance-Aware Delta-Sparsification for Improved Model Compression and Merging in LLMs](https://arxiv.org/abs/2504.13237) — `yang_impart_2025`. Assigns different sparsification ratios to singular vectors according to importance and combines with delta quantization and model merging. Especially close to MetaPAC's importance-to-compression allocation, while operating on fine-tuning deltas.

- [AdaRank: Adaptive Rank Pruning for Enhanced Model Merging](https://arxiv.org/abs/2503.22178) — `lee_adarank_2026`. Learns which singular components of task vectors to prune using test-time entropy minimization. Relevant adaptive low-rank allocation baseline for reducing merging interference; requires explicit accounting for adaptation data and compute.

## Added search terms

- model merging — `wortsman_model_soups_2022`
- weight-space averaging — `wortsman_model_soups_2022`
- model soups — `wortsman_model_soups_2022`
- same-task model merging — `wortsman_model_soups_2022`
- layer-wise weight averaging — `jang_model_stock_2024`
- geometric model merging — `jang_model_stock_2024`
- Fisher-weighted model merging — `matena_fisher_merging_2022`
- parameter-importance-weighted averaging — `matena_fisher_merging_2022`
- dataless knowledge fusion — `jin_regmean_2023`
- RegMean — `jin_regmean_2023`
- prediction-preserving weight fusion — `jin_regmean_2023`
- task arithmetic — `ilharco_task_arithmetic_2023`
- task vectors — `ilharco_task_arithmetic_2023`
- multi-task model merging — `ilharco_task_arithmetic_2023`
- task interference — `yadav_ties_merging_2023`
- sign-conflict resolution — `yadav_ties_merging_2023`
- sparse task-vector merging — `yadav_ties_merging_2023`
- delta-parameter sparsification — `yu_dare_2024`
- drop-and-rescale — `yu_dare_2024`
- DARE — `yu_dare_2024`
- adaptive model merging — `yang_adamerging_2024`
- layer-wise merging coefficients — `yang_adamerging_2024`
- entropy-minimization model merging — `yang_adamerging_2024`
- magnitude-based delta sampling — `deep_della_merging_2024`
- MAGPRUNE — `deep_della_merging_2024`
- outlier-filtered task vectors — `davari_model_breadcrumbs_2024`
- sparse-mask model merging — `davari_model_breadcrumbs_2024`
- task singular vectors — `gargiulo_task_singular_vectors_2025`
- low-rank task-vector compression — `gargiulo_task_singular_vectors_2025`
- subspace interference — `gargiulo_task_singular_vectors_2025`
- importance-aware delta sparsification — `yang_impart_2025`
- delta quantization — `yang_impart_2025`
- singular-vector sparsity allocation — `yang_impart_2025`
- adaptive rank pruning for model merging — `lee_adarank_2026`
- singular-direction selection — `lee_adarank_2026`

## Relevance

ImPart, AdaRank and TSV-Merge are priority adjacent methods for importance-guided delta sparsification, adaptive rank allocation and compression-aware merging. They are not asserted to be direct end-to-end mixed-compression competitors to MetaPAC. Model Soups and Task Arithmetic provide basic merging baselines; Fisher, TIES, DARE and DELLA provide alternative importance and interference treatments. These papers mainly study compatible models derived from a common pretrained initialization.

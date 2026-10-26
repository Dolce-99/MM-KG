# Aligning Multimodal Patient Evidence with Biomedical Knowledge Graphs for Clinical LLMs

Official implementation of **MM-KG** (paper under review).

<!-- [![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX) -->

<p align="center"><img src="figs/overview.png" width="85%"></p>

MM-KG keeps a patient's multimodal evidence (text, images, codes, measurements, …) and a
biomedical knowledge graph as two separate layers of one typed graph, joined by explicit
alignment edges. Each modality goes through its own *harmonizer* into typed, provenance-linked
observation nodes mapped to standard concepts; an *aligner* links those concepts to the knowledge
graph; *query-conditioned retrieval* selects a compact subgraph for a question; and the subgraph
is consumed by an LLM or a GNN. Because every alignment edge is an individually addressable
object, patient evidence, knowledge, and their correspondence can each be retrieved, traced,
withheld or corrupted — which is what the paper's factorial and disruption experiments measure.

### Why MM-KG?

* **Not fusion.** Fused multimodal representations cannot say which observation supported an
  answer, nor be ablated one source at a time. MM-KG can.
* **Knowledge as explicit links, not background text.** On questions that need both the patient
  and the knowledge graph, neither source alone is far above chance; their combination is, and
  deleting the single answer-bearing relation removes the gain.
* **Cheap retrieval.** The query-conditioned subgraph matches or beats generic vector / full-context
  retrieval at a fraction of the tokens.
* **A general backbone.** This repository instantiates the method on MIMIC-IV / MIMIC-CXR with
  PrimeKG; the paper's second instantiation (ADNI) uses the same contract with its own harmonizers.

## Citation

```
@article{du2026aligning,
  title={Aligning Multimodal Patient Evidence with Biomedical Knowledge Graphs for Clinical LLMs},
  author={Du, Jiawen and Khan, Arshan Ali and Zhang, Chenhao and Plotkin, Zachary and Shen, Li and Long, Qi and Li, Yun and Chen, Can and Chen, Tianlong and Konz, Nicholas},
  journal={arXiv preprint arXiv:2610.06685},
  year={2026}
}
```

## 1) Installation

Python ≥ 3.10. Install [PyTorch](https://pytorch.org) for your CUDA version, then

```bash
pip3 install -r requirements.txt
```

The scripts form three dependency groups, named in each script's docstring: `umls_eval`
(pandas, spaCy, medspaCy, QuickUMLS — CPU), `torch_gpu` (PyTorch, PyTorch Geometric,
transformers, open_clip) and `llm_qwen` (vLLM; a 48 GB GPU serves the 35B GPTQ model).
Keeping them in separate environments avoids version conflicts between vLLM and PyTorch Geometric.

## 2) Data

None of the data is redistributed. You need credentialed access to
[MIMIC-IV v3.1](https://physionet.org/content/mimiciv/), [MIMIC-CXR-JPG v2.1.0](https://physionet.org/content/mimic-cxr-jpg/)
and [MIMIC-IV-Note v2.2](https://physionet.org/content/mimic-iv-note/), the
[PrimeKG](https://github.com/mims-harvard/PrimeKG) release, a [UMLS](https://www.nlm.nih.gov/research/umls/)
Metathesaurus (`META/` RRF files), and the `mondo.sssom.tsv` mapping file from a
[Mondo](https://github.com/monarch-initiative/mondo) release placed next to the PrimeKG files.

Point the scripts at your copies with environment variables (defaults under `MMKG_HOME`):

| variable | default | content |
|---|---|---|
| `MIMIC_IV_DIR` | `$MMKG_HOME/data/mimiciv/3.1` | `hosp/*.csv.gz`, `icu/icustays.csv.gz` |
| `MIMIC_CXR_DIR` | `$MMKG_HOME/data/mimic-cxr-jpg/2.1.0` | `files/`, metadata and CheXpert label CSVs |
| `MIMIC_NOTE_DIR` | `$MMKG_HOME/data/mimic-iv-note/2.2/note` | `radiology.csv.gz`, `radiology_detail.csv.gz` |
| `PRIMEKG_DIR` | `$MMKG_HOME/data/primekg` | `kg.csv`, `nodes.tab`, `*_features.tab`, `mondo.sssom.tsv` |
| `UMLS_META_DIR` | `$MMKG_HOME/data/umls/2026AA/META` | `MRCONSO.RRF`, `MRREL.RRF`, `MRSAT.RRF`, `MRSTY.RRF` |
| `QUICKUMLS_INDEX` | `$MMKG_HOME/.cache/quickumls_index` | built once: `python -m quickumls.install -L -U -E ENG $UMLS_META_DIR $QUICKUMLS_INDEX` |
| `HF_HOME` | `$MMKG_HOME/.cache/huggingface` | Hugging Face cache; scripts run with `HF_HUB_OFFLINE=1` |

Models to download into `HF_HOME`: `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`,
`cambridgeltl/SapBERT-from-PubMedBERT-fulltext`, `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract`,
`emilyalsentzer/Bio_ClinicalBERT`, `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` (and, for the replications,
`Qwen/Qwen3.5-9B`, `meta-llama/Llama-3.1-8B-Instruct`).

## 3) Build the graph

Scripts are numbered in execution order and write to `<folder>/outputs/`. Steps marked *array*
process shards in parallel when `WG_NSHARDS` (shard count) and `SLURM_ARRAY_TASK_ID` (shard index)
are set, under any scheduler; unset, they run everything in one process.

```bash
cd graph
python 00_mrconso_subset.py                     # UMLS CUI/SAB/CODE projection for the bridge
python build_crosswalk.py                       # concept -> PrimeKG bridge tables
python 01_whole_cohort.py                       # cohort: chest film ∧ admission ∧ ≥1 diagnosis
python 02_biomedclip_findings.py                # imaging harmonizer (array)   -> 02b_merge_imaging.py
python 03a_prep_reports.py                      # report harmonizer: 03b_report_findings.py (array) -> 03c_merge_reports.py
python 04_patient_tables.py                     # patient nodes/edges + ALIGNS_TO_CUI alignment edges
python 05_build_pyg.py                          # + clinical PrimeKG; graph summary
python 06_sapbert_names.py; python 07_pubmedbert_desc.py; python 08_bioclinbert_reports.py   # node features (08 is an array) -> 08b
python 09_build_pyg_embed.py                    # PyG HeteroData with pretrained features
```

## 4) Run the experiments

**Patient-specific join questions and the 2×2 factorial** (patient evidence × knowledge graph):

```bash
cd tasks
python 00_build_rel_cache.py                    # PrimeKG relation cache
python 02a_medsafety_kg.py                      # relations of each patient's own drugs/conditions
cd ../graph && python 13_icu_ihm_features.py && cd ../tasks   # ICU cohort, 48 h window
python 18_build_composite_wide.py && python 23_prepare_v2_benchmark.py     # 3,200-item benchmark, matched negatives
python 19_composite_wide_llm.py --model Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 --tag widev2_q35 \
       --benchmark composite_benchmark_wide_v2.jsonl --max_model_len 8192         # arms A/B/C/D + leave-one-channel-out
python 19_composite_wide_llm.py --tag widev2_q35 --benchmark composite_benchmark_wide_v2.jsonl --analyze_only
```
Add `--arms C_alt D_alt --max_model_len 32768` for the typed-triple rendering, `--model Qwen/Qwen3.5-9B`
or `meta-llama/Llama-3.1-8B-Instruct` for the replications, and `26_subset_interactions.py` for the
subset analyses.

**Retrieval policies, candidate ranking, external baselines** (`retrieval/`; every script checks the
digests of its frozen inputs and logs its configuration under `retrieval/outputs/`):

```bash
cd retrieval && python stage0_audit/freeze_subsets_and_splits.py        # patient-level dev/test split
cd exp1_retrieval
python 01_build_evidence_collection.py && python 02_build_kg_onehop_cache.py
python 03_build_contexts.py --policy R4 --split test --budget natural --tag R4_test_bnatural   # R0..R5; --kg_order round_robin, --budget match_r4|2048|4096
python 05_score_contexts.py --model q35 --max_model_len 8192 --tags R4_test_bnatural ...
python 06_analyze.py --model q35 --split test --tags ... --ref R4_test_bnatural
cd ../exp2_ranking
python 01_build_ranking_benchmark.py --max_total_dist 12
python 02_build_contexts.py --version ranking_v1_d12 --split test --arm D --policy R4q --control none   # arms A-D, controls, channel ablations
python 03_score_ranking.py --model q35 --max_model_len 8192 --tags ...; python 04_analyze.py --version ranking_v1_d12 --model q35 --split test --tags ...
cd ../exp4_baselines                             # GraphCare (03/04/07, 08/09) and MindMap (05/06, 10), analysis 11/12
```

**Fact-checking, knowledge-graph validation, outcome prediction:**

```bash
cd tasks
python 09a_build_audit_cohort.py && python 06a_build_extended_benchmark.py --complete
python 06_extended_audit_llm.py --model Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 --benchmark extended_benchmark_complete.jsonl --tag complete --max_model_len 16384
python 07_external_validation.py && python 08a_refine_contraindications.py          # prescribing-behaviour validation of PrimeKG
python 12_build_mm_bundles.py && python 13_mm_ablation_llm.py --tag ihm            # LLM ICU-mortality channel ablation
cd ../graph && WG_FINETUNE=1 python 13_icu_ihm_whole.py && python 13c_gnn_paired.py   # GNN ICU-mortality channel ablation
```

## 5) Use MM-KG on your own data

The construction, retrieval and consumer code depend only on the graph schema, not on MIMIC or
PrimeKG. To instantiate the method elsewhere: (1) write harmonizers that emit the node/edge
tables of `graph/04_patient_tables.py` (concept-keyed observation nodes, one edge type per
derivation route, provenance in keys and attributes); (2) build bridge tables from your concept
namespace to your knowledge graph's node indices (`graph/build_crosswalk.py`); (3) define the
question families' relation layers and the occult-state rules of your cohort (`LAYER` and
`LAB_STATES` in `tasks/18_build_composite_wide.py`). Everything downstream runs unchanged.

## Troubleshooting

* **vLLM cannot find `nvcc` / CUDA.** On hosts without `/usr/local/cuda`, point `CUDA_HOME` at the
  toolkit shipped with the pip wheels (`<env>/lib/python3.11/site-packages/nvidia/cu13`) or set
  `VLLM_USE_FLASHINFER_SAMPLER=0`.
* **AUROC exactly 0.500 in every arm.** The model never emitted a yes/no token (reasoning models
  must be run with thinking disabled; the scripts do this). Check the "without a yes/no token"
  count the scorer prints.
* **Comparing arms across jobs.** Scores from different engine instances differ by up to ±0.004
  AUROC; score all arms of one comparison in one process, as the scripts do.

## Acknowledgements

PrimeKG (Chandak, Huang & Zitnik, 2023), MIMIC-IV / MIMIC-CXR / MIMIC-IV-Note (PhysioNet), UMLS,
the Mondo Disease Ontology, and the open models listed above. `retrieval/exp4_baselines/graphcare_model.py`
is adapted from the [GraphCare](https://github.com/pat-jj/GraphCare) authors' code (MIT License);
MindMap is re-implemented from the paper. This code is a research artifact, not a clinical decision tool.

# Empatia_DL

**Physiological emotion recognition using two-dimensional feature maps and deep learning.**

Empatia_DL provides research code for transforming handcrafted physiological descriptors into 2D feature maps and learning emotion-related patterns with convolutional neural networks. It includes WEMAC and WESAD workflows, feature-wise normalization, feature-map augmentation, and Fisher-guided recursive feature elimination.

This repository accompanies:

> Junjiao Sun, Jorge Portilla, and Andres Otero. **A Deep Learning Approach for Fear Recognition on the Edge Based on Two-Dimensional Feature Maps.** *IEEE Journal of Biomedical and Health Informatics*, 28(7), 3973–3984, 2024. [DOI: 10.1109/JBHI.2024.3392373](https://doi.org/10.1109/JBHI.2024.3392373).

It supports the 2D representation and CNN-based modeling work in **Chapter 3** of Junjiao Sun's doctoral thesis, *Toward Practical and Uncertainty-Aware Affective Computing with Wearable Physiological Signals*. The chapter's CNN–LSTM extension is maintained separately in [Empatia_CNN-LSTM](https://github.com/des-cei/Empatia_CNN-LSTM).

> **Implementation status:** This repository contains research scripts with editable paths and experiment settings. Raw datasets, generated feature caches, normalization logs, and pretrained checkpoints are not included. Running the full workflow requires the compatibility adjustments below. The checked-in WESAD configuration differs from the wrist-signal configuration described in the thesis.

## Method overview

The pipeline connects physiological feature engineering with CNN-based representation learning:

1. Extract window-level cardiovascular, electrodermal, and temperature descriptors.
2. Apply feature-wise normalization (FWN) to accommodate different feature scales and distributions.
3. Organize descriptors from successive windows into 2D feature maps.
4. Augment the maps by combining label-compatible feature frames.
5. Train CNN classifiers, optionally reducing the feature set through Fisher-guided recursive feature elimination (RFE).

The feature modules implement **123 descriptors**: 84 cardiovascular/BVP descriptors, 34 GSR descriptors, and 5 skin-temperature descriptors. They include pulse morphology and variability measures, electrodermal statistics, and temperature summaries.

The current generators use `numpy.resize` to produce square maps of shape `(F, F)`, where `F` is the retained feature count. With all descriptors, this is `123 × 123`. Maps are stacked as `(F, F, N)` and converted to `(N, 1, F, F)` for PyTorch. These are numerical feature maps, not RGB images. `numpy.resize` repeats or truncates flattened values rather than interpolating an image; this implementation should be distinguished from the conceptual descriptor-by-time matrix in the thesis.

| Strategy | Implementation role |
| --- | --- |
| `AllFromOne` | Build maps from one participant's data, grouped by video in WEMAC and by class in WESAD. |
| `HalfAndHalf` | Combine the current sequence with a randomly selected, label-compatible participant sequence. |
| `HalfAndRandom` | Combine the current sequence with randomly sampled, label-compatible frames. |
| `All_concat` | Training-script option that combines the maps and labels from all three strategies. |

Mixing is dataset-specific: WEMAC interleaves frames, while WESAD uses block/frame sampling. The names describe the augmentation idea, not a guaranteed 50/50 split after resizing.

## Repository guide

| Location | Purpose |
| --- | --- |
| `WEMAC/Pack_all_data.py` | Extract features from MATLAB recordings and associate spreadsheet labels. |
| `WEMAC/Transfer_label.py` | Convert the older IT06 label format when needed. |
| `WEMAC/Data_extraction.py` | Read cached features, split participants, and provide outlier handling. |
| `WEMAC/Data_normalization.py` | Search for and apply feature-wise normalization. |
| `WEMAC/Create_feature_maps.py` | Construct and augment WEMAC feature maps. |
| `WESAD/read_data.py` | Load WESAD subject pickle files. |
| `WESAD/Feature_extraction_wesad.py` | Extract features into per-subject caches. |
| `WESAD/Abnormal_extraction_weasd.py` | Load caches, retain target classes, and prepare features; this is the actual filename. |
| `WESAD/Data_normalization_WESAD.py` | Search for and apply WESAD normalization. |
| `WESAD/Create_feature_maps_wesad.py` | Construct and augment WESAD feature maps. |
| Each dataset's `*_Signal_Features.py` files | Physiological descriptor implementations. |
| Each dataset's `FWN_main/` directory | Bundled normalization and optimization routines. |
| Each dataset's `Training/` directory | Classification and feature-selection experiments. |
| Each `Training/models/` directory | CNNs including VGG, ResNet, GoogLeNet, DenseNet, EfficientNet, MobileNet, and ResNeXt. |

## Environment and installation

The original repository records this training environment:

| Component | Recorded configuration |
| --- | --- |
| GPU | NVIDIA A30 |
| CUDA environment | 12.2 |
| NVIDIA driver | 535.154.05 |
| PyTorch | `2.0.0+cu118` |

The CUDA environment version and the PyTorch CUDA build tag are separate values. Use a compatible PyTorch/torchvision pair and GPU driver. A Python version and a pinned dependency lockfile are not specified.

Create an isolated Python environment and install PyTorch and torchvision appropriate for your machine. Additional dependencies inferred from the source imports include:

```bash
python -m pip install numpy scipy pandas scikit-learn matplotlib seaborn \
    mat73 biosppy neurokit2 distfit xlrd xlwt skfeature-chappers
```

This is an import-derived package list, not a tested, version-locked environment. `Transfer_label.py` uses legacy `.xls` writing, which requires a compatible pandas writer or adaptation to another format. The Fisher helper expects a `skfeature` implementation supporting `mode='rank'`.

The training loops explicitly cast inputs to `torch.cuda.FloatTensor`, so the current scripts require CUDA despite containing a CPU device-selection branch. CPU execution requires replacing these casts with device-aware conversions such as `inputs.to(device=device, dtype=torch.float32)`.

## Preparation required before running

Clone the repository and expose its local modules in a Bash terminal:

```bash
git clone https://github.com/des-cei/Empatia_DL.git
cd Empatia_DL
export EMPATIA_ROOT="$PWD"
export PYTHONPATH="$EMPATIA_ROOT:$EMPATIA_ROOT/WEMAC:$EMPATIA_ROOT/WESAD${PYTHONPATH:+:$PYTHONPATH}"
```

The following workflows assume this environment. Apply these adjustments in your local copy first:

| Item | Required adjustment |
| --- | --- |
| Local paths | Update hard-coded dataset paths and obsolete `sys.path.append(...)` entries. Relative paths depend on the working directory. |
| Missing WESAD utility import | `WESAD/Training/utils.py` imports the absent `Feature_wise_Normalization.Create_feature_maps`. Remove this unused import for training, or adapt its standalone example to the included WESAD helpers. |
| JSON writers | Both exporters append `",\n"` to serialized objects, while readers use `json.load`. Write one valid object per cache with `json.dump(..., cls=MyEncoder)` in write mode. Regenerate caches rather than appending objects. |
| WESAD output paths | Create `WESAD/json_files/`, use `os.path.join` instead of backslash concatenation, and keep only per-subject caches there. Store the optional aggregate cache elsewhere. |
| WEMAC normalization log | The writer defaults to `log_data_normalization_functions_svm_AD.log`, while consumers expect `log_data_normalization_functions_KNN_AD.log`. Use a consistent filename. The active search classifier is KNN. |
| WESAD map demo log | The standalone generator points to a WEMAC log. Change it to `log_data_normalization_WESAD_KNN.log`. |

Installing dependencies alone does not resolve these source-level issues.

## WEMAC workflow

### 1. Obtain and configure the data

Obtain WEMAC through the [dataset authors' access page](https://www.uc3m.es/institute-gender-studies/DATASETS) and consult the [dataset description](https://doi.org/10.1038/s41597-024-04002-8).

The default extractor expects `BBDDLab_EH_CEI_VVG_IT07.mat` and `Labels_TabLab_IT07.xls` in `WEMAC/`. These refer to the development-time MATLAB/spreadsheet layout. Adapt the loader if your release differs. It reads the MATLAB key `BBDDLab_EH_CEI_VVG_IT07` and preprocessed fields `bvp_filt`, `gsr_uS_filtered_sm`, and `skt_filt_dn_sm`.

For the older IT06 format, inspect `Transfer_label.py` and update the corresponding filenames, label handling, and MATLAB key.

### 2. Extract and normalize features

After completing the preparation adjustments:

```bash
cd "$EMPATIA_ROOT/WEMAC"
python Pack_all_data.py
python Data_normalization.py
```

BVP and GSR are sampled at 200 Hz and skin temperature at 10 Hz. The extractor uses 3,000-sample BVP/GSR windows (15 seconds), with 2,500-sample overlap; temperature indexing is adjusted to its sampling rate.

`Pack_all_data.json` stores participant-, trial-, and window-level feature dictionaries. Training uses `ReportadaBinarizado`. Normalization produces a log containing the selected function for each feature. The downstream parser requires the `Over all Best N:` entry, so an incomplete log is insufficient.

The default normalization search is substantial: 100 repeated searches, each with population 10 and 100 optimization iterations.

### 3. Inspect feature maps and train

```bash
python Create_feature_maps.py
cd Training
python General_F1.py --lr 0.1
```

Map generation creates arrays in memory; it does not automatically export image files. Training invokes these functions directly.

| Script | Experiment |
| --- | --- |
| `General_F1.py` | Repeated random participant-level 80/20 splits for fear/non-fear classification. |
| `LASO_F1.py` | Legacy partial-subject holdout experiment. |
| `train_rfe_fisher.py` | Fisher-guided RFE with repeated classifier training. |

`LASO_F1.py` holds out only part of the selected participant's data. Other data from that participant can remain in training, so it is **not strict leave-one-subject-out evaluation**.

## WESAD workflow

### 1. Obtain and configure the data

Download [WESAD from UCI](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection). The reader expects files such as `<dataset-root>/S2/S2.pkl`. Set the root passed to `chest_data_label(...)` in `Feature_extraction_wesad.py` to an absolute path. Use absolute output paths too, because the reader changes the working directory when loading subjects.

**Current signal configuration:** The extractor uses chest `ECG`, `EDA`, and `Temp` at 700 Hz, with 6,000-sample windows and 4,000-sample overlap. ECG is passed through the cardiovascular feature module under BVP-style names. This differs from the thesis's wrist BVP/GSR/SKT configuration. Reproducing the wrist configuration requires changing signal selection, windowing, and label alignment.

### 2. Extract and normalize features

After correcting JSON output and path handling:

```bash
cd "$EMPATIA_ROOT/WESAD"
mkdir -p json_files
python Feature_extraction_wesad.py
python Data_normalization_WESAD.py
```

The loader expects per-subject caches such as `json_files/Pack_all_data_WESAD_S2.json`, containing window-keyed feature dictionaries and a `label` field. The normalization output is `log_data_normalization_WESAD_KNN.log`.

Original WESAD labels 1, 2, and 3 are retained and remapped:

| Condition | Three-class label | Two-class label |
| --- | --- | --- |
| Baseline | 0 | 0: non-stress |
| Stress | 1 | 1: stress |
| Amusement | 2 | 0: non-stress |

### 3. Inspect feature maps and train

```bash
python Create_feature_maps_wesad.py
cd Training
python General_WESAD_F1.py --lr 0.1
```

The checked-in `AllFromOne` generator uses blocks of 60 **feature frames**, advanced by 15 frames. These values are frame counts, not seconds.

| Task | General training | Subject holdout | Feature selection |
| --- | --- | --- | --- |
| Three classes | `General_WESAD_F1.py` | `LOSO_WESAD_F1.py` | `train_rfe_wesad_fisher_3C.py` |
| Two classes | `General_WESAD_F1_2class.py` | `LOSO_WESAD_F1_2class.py` | `train_rfe_wesad_fisher_2C.py` |

Run the chosen script from `WESAD/Training/`. The LOSO scripts call `LOSO_resplit` to hold out a subject. The alternative `LOSO_resplit_new` helper retains all samples in training and must not be used for subject-independent evaluation.

## Training settings and outputs

| Setting | Configuration location |
| --- | --- |
| CNN architecture | `model_group` in the main block. |
| Augmentation | `strategy_list` or `strategy`. |
| Repetitions | `K` in the experiment loop. |
| Epochs | Epoch loop inside the training function. |
| Batch size | `DataLoader` construction, generally 32. |
| Learning rate | `--lr`, default `0.1`. |
| RFE stopping point | `min_feature_num`; check whether metadata rows are counted. |

Current training functions use cross-entropy loss, SGD with momentum `0.9` and weight decay `5e-4`, and cosine learning-rate scheduling. These repository defaults differ from the general training settings described in the thesis.

For an initial run, reduce the main block to one architecture, `AllFromOne`, and one repetition. The default loops can launch many experiments.

Scripts print loss, accuracy, weighted F1-score, and aggregate statistics. Accuracy and F1 are printed as percentages. Console output can be retained with:

```bash
python General_F1.py --lr 0.1 > training.log 2>&1
```

Substitute the appropriate script for WESAD. Checkpoint saving is commented out in the main workflows; a completed run does not automatically produce a reusable model file. The `--resume` option does not establish a complete save/resume workflow.

## Evaluation and reproducibility

The public scripts preserve development-time experiment logic. Before reporting independent benchmark results, account for these implementation details:

- **Splits and initialization:** General scripts repeat random participant splits, not conventional fixed K-fold partitions. Several loops reuse instantiated models across repetitions, strategies, or subjects. Use a fresh model for every independent run and record random seeds and subject assignments.
- **Normalization and feature selection:** Normalization occurs before LOSO re-splitting, and normalization search selects configurations using the split's prediction error. RFE ranks features using concatenated partitions and uses evaluation performance for removal decisions. For unbiased testing, perform all selection within training subjects and an inner validation split, leaving the final test subject untouched.
- **Augmentation:** Training functions apply the selected strategy to both partitions. To evaluate on original samples, explicitly use `AllFromOne` for the held-out partition.
- **Epoch selection:** Best accuracy and best F1 are selected across evaluation epochs and may come from different epochs. For final testing, select a checkpoint on validation data and evaluate once on the held-out set.
- **Incomplete recordings:** Some augmentation branches assume enough frames exist for every requested participant/class or video. Missing groups and short recordings need explicit checks before indexing or random sampling.

Default execution alone therefore does not establish reproduction of the thesis's final evaluation protocol or scores. Refer to the associated paper for its experimental results and record the exact configuration of any new run. The repository contains CNN experimentation code, but no end-to-end Coral Edge TPU conversion and deployment pipeline.

## Citation

If you use this code or the feature-map methodology, please cite:

```bibtex
@article{sun2024fear,
  author  = {Sun, Junjiao and Portilla, Jorge and Otero, Andres},
  title   = {A Deep Learning Approach for Fear Recognition on the Edge Based on Two-Dimensional Feature Maps},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  year    = {2024},
  volume  = {28},
  number  = {7},
  pages   = {3973--3984},
  doi     = {10.1109/JBHI.2024.3392373}
}
```

Please also cite the original WEMAC or WESAD dataset publication when using its data.

## License

No repository-level license file is currently included. Contact the maintainers to clarify reuse and redistribution terms. Dataset access and reuse remain subject to the respective providers' terms.

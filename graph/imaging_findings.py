"""
imaging_findings.py   --   stdlib only (imports cleanly in torch_gpu AND umls_eval)
==================================================================================
Single source of truth for the BioMedCLIP imaging-finding vocabulary shared by
step 02 (runs the classifier, env torch_gpu) and step 03 (turns present findings
into IMAGING_FINDING -> Finding(CUI) edges, env umls_eval).

BioMedCLIP is a fixed 13-label CheXpert zero-shot classifier (not a text
generator): for each finding it contrasts a positive vs negative prompt and
softmaxes, giving P(present). We map each label to ONE canonical disorder CUI so
the imaging finding lands on the SAME Finding(CUI) node as the matching ICD
diagnosis (cross-modal unification) and, where possible, aligns into PrimeKG.

CUI choices were validated against the report extractor (nlp_cui / QuickUMLS) and
the PrimeKG bridge (cui2primekg.tsv): 8 of 13 align to PrimeKG; the other 5 are
radiographic/device concepts with no disease-KG node and are kept patient-side
(no ALIGNS_TO_CUI edge), per the keep-all rule.
"""
from __future__ import annotations

# CheXpert-13 (drops "No Finding"; it is the negative anchor, never a positive edge)
CHEXPERT_FINDINGS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
    "Pleural Effusion", "Pleural Other", "Pneumonia", "Pneumothorax",
    "Support Devices",
]

# Readable phrase per label for the CLIP pos/neg text prompts (from phase-1 01b).
LABEL_PHRASE = {
    "Atelectasis": "atelectasis",
    "Cardiomegaly": "cardiomegaly",
    "Consolidation": "consolidation",
    "Edema": "pulmonary edema",
    "Enlarged Cardiomediastinum": "an enlarged cardiomediastinum",
    "Fracture": "a fracture",
    "Lung Lesion": "a lung lesion",
    "Lung Opacity": "a lung opacity",
    "Pleural Effusion": "pleural effusion",
    "Pleural Other": "a pleural abnormality",
    "Pneumonia": "pneumonia",
    "Pneumothorax": "pneumothorax",
    "Support Devices": "a support device",
}

# label -> (CUI, display name).  aligns=whether the CUI reaches PrimeKG (comment only)
LABEL_CUI = {
    "Atelectasis":                ("C0004144", "atelectasis"),            # aligns
    "Cardiomegaly":               ("C0018800", "cardiomegaly"),          # aligns
    "Consolidation":              ("C0521530", "lung consolidation"),    # patient-side only
    "Edema":                      ("C0034063", "pulmonary edema"),       # aligns
    "Enlarged Cardiomediastinum": ("C0746459", "mediastinal widening"),  # patient-side only
    "Fracture":                   ("C0016658", "fracture"),              # aligns
    "Lung Lesion":                ("C0577916", "lung lesion"),           # patient-side only
    "Lung Opacity":               ("C2073563", "lung opacity"),          # patient-side only
    "Pleural Effusion":           ("C0032227", "pleural effusion"),      # aligns
    "Pleural Other":              ("C0264545", "pleural thickening"),    # aligns
    "Pneumonia":                  ("C0032285", "pneumonia"),             # aligns
    "Pneumothorax":               ("C0032326", "pneumothorax"),          # aligns
    "Support Devices":            ("C0183683", "support device"),        # patient-side only (device)
}

PRESENT_THRESHOLD = 0.5   # P(present) >= this -> emit an IMAGING_FINDING edge

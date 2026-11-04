"""Three-category source-proxy batching and observed-cell-only loss.

No model is loaded, fitted or scored. Mathematical loss support grants no data
readiness: the default source loader retains the adapter's training gate.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
import torch
import torch.nn.functional as F
from adapt_function_source_labels import VERSION as SOURCE_VERSION, DEFINITION, masked_source_example
from audit_function_pilot_handoff import jsonl_rows, sha256
from prepare_context_review import safe_file, verify_artifacts
from prepare_function_review import CATEGORIES

VERSION = 'function-partial-source-loss-v1'
INPUT_PROTOCOL = 'post-statement-and-if-block-inline-audit-tail-v2'
MODEL_FIELDS = {'record_id','model_code','model_code_sha256','input_protocol'}


@dataclass(frozen=True)
class SourceBatch:
    record_ids: tuple[str, ...]
    model_codes: tuple[str, ...]
    targets: torch.Tensor
    observation_mask: torch.Tensor
    for_training: bool
    categories: tuple[str, ...] = CATEGORIES
    target_definition_id: str = DEFINITION


@dataclass(frozen=True)
class MaskedLoss:
    loss: torch.Tensor | None
    observed_cells: int
    observed_rows: int
    observed_per_category: tuple[int, ...]

    @property
    def must_skip(self) -> bool:
        """An unobserved batch must not step an optimizer, even with weight decay."""
        return self.loss is None


def collate_source_examples(pairs, *, for_training=True):
    """Return code-only inputs and inseparable target/mask tensors on CPU.

    Diagnostic purpose is explicit and never makes examples training eligible.
    SourceProxy metadata/rationales are not appended to model code.
    """
    if type(for_training) is not bool:raise ValueError('Purpose must be explicit boolean')
    pairs = list(pairs)
    if not pairs:raise ValueError('Empty batch')
    identities=[];codes=[];values=[];masks=[]
    for model,target in pairs:
        if set(model) != MODEL_FIELDS or model['input_protocol'] != INPUT_PROTOCOL:
            raise ValueError('Require the pinned code-only input protocol')
        code,labels,mask = masked_source_example(model,target,for_training=for_training)
        if model['record_id'] in identities:raise ValueError('Duplicate batch record ID')
        identities.append(model['record_id']);codes.append(code);values.append(labels);masks.append(mask)
    return SourceBatch(tuple(identities),tuple(codes),torch.tensor(values,dtype=torch.float32),
                       torch.tensor(masks,dtype=torch.bool),for_training)


def load_source_batch(source_release, *, expected_manifest_sha256, for_training=True):
    """Read one explicitly hash-pinned release; no model/download/training call."""
    root=Path(source_release).resolve()
    if not isinstance(expected_manifest_sha256,str) or not re.fullmatch('[0-9a-f]{64}',expected_manifest_sha256):
        raise ValueError('An explicit expected manifest SHA-256 is required')
    manifest_path=safe_file(root,'manifest.json')
    if sha256(manifest_path) != expected_manifest_sha256:raise ValueError('Source release identity mismatch')
    manifest=verify_artifacts(root,'manifest.json','artifact_sha256')
    if (manifest['protocol'] != SOURCE_VERSION or manifest['target_definition_id'] != DEFINITION
        or set(manifest['model_input_fields']) != MODEL_FIELDS):
        raise ValueError('Source-proxy schema differs; never infer a legacy eleven-output mapping')
    models=list(jsonl_rows(safe_file(root,'model_inputs.jsonl')))
    rows=list(jsonl_rows(safe_file(root,'source_targets.jsonl')))
    targets={r['record_id']:r for r in rows}
    if len(targets)!=len(rows) or len(models)!=len(rows) or {m['record_id'] for m in models}!=set(targets):
        raise ValueError('Input and target inventories differ or duplicate IDs')
    result=collate_source_examples(((m,targets[m['record_id']]) for m in models),for_training=for_training)
    if sha256(manifest_path)!=expected_manifest_sha256:raise ValueError('Manifest changed during batching')
    verify_artifacts(root,'manifest.json','artifact_sha256')
    return result


def observed_bce_with_logits(logits, targets, observation_mask, *, categories,
                             positive_class_weights=None):
    """Mean weighted BCE over observed cells only; no synthetic negative cells.

    Class order must be declared explicitly. Default weights are one, with no
    automatic estimate from missing labels. Optional positive weights must be
    fixed from justified training data only. Never fit weights on evaluation.
    This mathematical primitive does not certify supervision/evaluation gates.
    """
    if tuple(categories) != CATEGORIES:raise ValueError('Exact three-category order required')
    if not all(isinstance(t,torch.Tensor) for t in (logits,targets,observation_mask)):
        raise ValueError('Tensor inputs required')
    if logits.ndim!=2 or logits.shape[1]!=len(CATEGORIES) or targets.shape!=logits.shape or observation_mask.shape!=logits.shape:
        raise ValueError('Expected aligned batch-by-three tensors, not legacy eleven outputs')
    if not logits.is_floating_point() or not targets.is_floating_point() or targets.dtype!=logits.dtype:
        raise ValueError('Matching floating logits and target dtypes required')
    if observation_mask.dtype!=torch.bool or targets.requires_grad:
        raise ValueError('Boolean observation mask and constant targets required')
    if targets.device!=logits.device or observation_mask.device!=logits.device:
        raise ValueError('Aligned tensor devices required')
    weights=None
    if positive_class_weights is not None:
        w=positive_class_weights
        if (not isinstance(w,torch.Tensor) or w.shape!=(len(CATEGORIES),)
            or w.dtype!=logits.dtype or w.device!=logits.device or w.requires_grad
            or not torch.isfinite(w).all().item() or not (w>0).all().item()):
            raise ValueError('Require three fixed finite positive class weights on the logits device/dtype')
        weights=w.expand_as(logits)[observation_mask]
    per_category=tuple(int(x) for x in observation_mask.sum(dim=0).tolist())
    cells=sum(per_category);rows=int(observation_mask.any(dim=1).sum().item())
    if cells==0:return MaskedLoss(None,0,0,per_category)
    # Select before BCE: multiplying an unmasked NaN/Inf loss by zero is not safe.
    observed_logits=logits[observation_mask];observed_targets=targets[observation_mask]
    if (not torch.isfinite(observed_logits).all().item() or not torch.isfinite(observed_targets).all().item()
        or not ((observed_targets==0)|(observed_targets==1)).all().item()):
        raise ValueError('Observed logits must be finite and observed labels explicit zero/one')
    loss=F.binary_cross_entropy_with_logits(observed_logits,observed_targets,pos_weight=weights,reduction='sum')/cells
    if not torch.isfinite(loss).item():raise ValueError('Nonfinite observed objective; no optimizer step allowed')
    return MaskedLoss(loss,cells,rows,per_category)

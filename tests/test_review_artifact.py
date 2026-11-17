"""Software-only publication-boundary tests; no research examples or labels."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
if (ROOT/'scripts/build_anonymous_artifact.py').exists():
 path=ROOT/'scripts/build_anonymous_artifact.py'
else:
 path=Path(__file__).with_name('build_anonymous_artifact.py')
spec=importlib.util.spec_from_file_location('review_builder',path)
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)

class ReviewArtifactBoundaries(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
 def test_existing_output_is_preserved(self):
  out=self.root/'existing';out.mkdir();p=out/'sentinel';p.write_text('keep')
  with self.assertRaises(FileExistsError):b.build(out,self.root/'missing',self.root/'repo',self.root/'paper')
  self.assertEqual(p.read_text(),'keep')
 def test_unknown_numeric_slice_is_not_copied(self):
  n=self.root/'untrusted.json';n.write_text('{}');out=self.root/'new'
  with self.assertRaises(ValueError):b.build(out,n,self.root/'repo',self.root/'paper')
  self.assertFalse(out.exists())
 def test_corpus_like_csv_is_rejected(self):
  p=self.root/'x.csv';p.write_text('contract_id,y_true,y_score\nexample,1,0.9\n')
  with self.assertRaises(ValueError):b.validate_aggregate_csv(p)
 def test_unknown_csv_field_fails_closed(self):
  p=self.root/'x.csv';p.write_text('dataset,unexpected_payload\nfixture,hidden\n')
  with self.assertRaises(ValueError):b.validate_aggregate_csv(p)
 def test_duplicate_csv_header_fails_closed(self):
  p=self.root/'x.csv';p.write_text('dataset,roc_auc,roc_auc\nfixture,hidden,0.5\n')
  with self.assertRaises(ValueError):b.validate_aggregate_csv(p)
 def test_oversized_free_text_is_rejected(self):
  p=self.root/'x.csv';p.write_text('dataset,roc_auc\n'+('x'*257)+',0.5\n')
  with self.assertRaises(ValueError):b.validate_aggregate_csv(p)
 def test_private_identifier_is_rejected(self):
  p=self.root/'x.csv';p.write_text('dataset,roc_auc\nuser@example.invalid,0.5\n')
  with self.assertRaises(ValueError):b.validate_aggregate_csv(p)
 def test_aggregate_csv_passes_without_dataset_code(self):
  p=self.root/'x.csv';p.write_text('dataset,model,roc_auc\nfixture,et,0.5\n');b.validate_aggregate_csv(p)
 def test_metadata_drops_case_and_local_path_payloads(self):
  x={'model':'et','feature_count':244,'cases':[{'not_for_export':'private'}],'local_path':'private'}
  self.assertEqual(b.metadata_projection('shap_results/primary/metadata.json',x),{'model':'et','feature_count':244})
 def test_manifest_drops_contract_identifiers(self):
  x={'scored_csv_sha256':'abc','dappscan_duplicate_contract_ids':['private']}
  self.assertEqual(b.metadata_projection('manifest.json',x),{'scored_csv_sha256':'abc'})
 def test_missing_metadata_field_fails_closed(self):
  with self.assertRaises(KeyError):b.metadata_projection('shap_results/primary/metadata.json',{'model':'et'})
 def test_symlink_source_is_rejected(self):
  p=self.root/'source';p.write_text('plain');(self.root/'alias').symlink_to(p)
  with self.assertRaises(ValueError):b.source_file(self.root,'alias')
 def test_tex_escape_is_rejected(self):
  root=self.root/'paper';root.mkdir();(self.root/'outside.tex').write_text('outside');(root/'sc_vulnerability_detection.tex').write_text(r'\input{../outside}')
  with self.assertRaises(ValueError):b.manuscript_closure(root)
 def test_missing_manuscript_dependency_fails(self):
  (self.root/'sc_vulnerability_detection.tex').write_text(r'\includegraphics{missing.pdf}')
  with self.assertRaises(ValueError):b.manuscript_closure(self.root)
 def test_literal_manuscript_closure_is_complete(self):
  (self.root/'sc_vulnerability_detection.tex').write_text(r'\input{section}\bibliography{refs}\includegraphics{figure.pdf}')
  for name in ['section.tex','refs.bib','figure.pdf']:(self.root/name).write_text('fixture')
  self.assertEqual(b.manuscript_closure(self.root),['figure.pdf','refs.bib','sc_vulnerability_detection.pdf','sc_vulnerability_detection.tex','section.tex'])
if __name__=='__main__':unittest.main()

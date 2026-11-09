"""Verify release hashes and a deterministic CPU inference fixture."""
import hashlib,json,subprocess,sys,tempfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'manifest_sha256.json').read_text())
for name,expected in manifest.items():
    actual=hashlib.sha256((root/name).read_bytes()).hexdigest()
    assert actual==expected, name
with tempfile.TemporaryDirectory() as tmp:
    out=Path(tmp)/'prediction.json'
    subprocess.run([sys.executable,str(root/'metcompose.py'),'predict','--checkpoint',str(root/'checkpoints/fold0_seed17'),'--smiles','NC(=O)C1CCCCC1','--output',str(out)],check=True)
    actual=json.loads(out.read_text());expected=json.loads((root/'tests/prediction_fixture.json').read_text())
    assert actual==expected, 'Prediction differs from release fixture; check dependency versions.'
print(f'PASS: {len(manifest)} file hashes and deterministic inference fixture')

/* ── openMIP ONNX identification pipeline ── */
// Depends on: onnxruntime-web (ort global), featurizePDB (predict.js),
//             rawPdbText (viewer state, set by viewer.html).

let identOrtSession = null;

// Load (or return the cached) ONNX session for identification.
async function loadIdentifyModel() {
  if (identOrtSession) return identOrtSession;
  identOrtSession = await ort.InferenceSession.create('src/identify-model.onnx');
  return identOrtSession;
}

// Standard amino acid three-letter codes used for residue counting.
const IDENTIFY_AA_NAMES = new Set([
  'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
  'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL',
]);

// Count distinct amino acid residues in a PDB text.
// A residue is identified by its chain + sequence number + insertion code.
function countAAResidues(pdbText) {
  const seen = new Set();
  for (const line of pdbText.split('\n')) {
    const rec = line.substring(0, 6).trim();
    if (rec !== 'ATOM' && rec !== 'HETATM') continue;
    const resName = line.substring(17, 20).trim();
    if (!IDENTIFY_AA_NAMES.has(resName)) continue;
    const key = line[21] + line.substring(22, 27); // chain + resSeq + iCode
    seen.add(key);
  }
  return seen.size;
}

// Show a transient toast notification (auto-dismisses after `duration` ms).
let _toastTimer = null;
function showToast(msg, duration = 4000) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.classList.add('visible');
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => el.classList.remove('visible'), duration);
}

// Populate the scrollable results list and reveal the Download PDB button.
// results: Array of { value: string, label: string }
function populateIdentifyResults(results) {
  const scroll = document.getElementById('identify-scroll');
  scroll.innerHTML = '';
  results.forEach(item => {
    const el = document.createElement('div');
    el.className = 'identify-scroll-item';
    el.dataset.value = item.value;
    el.textContent = item.label;
    el.addEventListener('click', () => {
      scroll.querySelectorAll('.identify-scroll-item')
            .forEach(i => i.classList.remove('selected'));
      el.classList.add('selected');
    });
    scroll.appendChild(el);
  });
  scroll.style.display = '';
  document.getElementById('prune-btn').style.display = '';
  document.getElementById('download-identify').style.display = '';
}

// Run the full identification pipeline for the currently loaded molecule.
// Reads rawPdbText from the viewer scope.
async function runIdentification() {
  const btn = document.getElementById('identify-btn');

  // Reset previous output.
  document.getElementById('identify-scroll').style.display = 'none';
  document.getElementById('prune-btn').style.display = 'none';
  document.getElementById('download-identify').style.display = 'none';

  if (!rawPdbText) {
    showToast('No molecule loaded.');
    return;
  }

  const aaCount = countAAResidues(rawPdbText);

  if (aaCount < 25) {
    // Covers both small molecules (aaCount === 0) and short epitopes.
    showToast('Epitope identification is only necessary on full proteins.');
    return;
  }

  btn.textContent = 'Running…';
  btn.classList.add('running');
  btn.disabled = true;

  try {
    const session  = await loadIdentifyModel();
    // Reuse the same featurizer as the prediction pipeline.
    const features = featurizePDB(rawPdbText);

    const inputName   = session.inputNames[0];
    const inputTensor = new ort.Tensor('float32', features, [1, features.length]);
    const results     = await session.run({ [inputName]: inputTensor });

    const outputName = session.outputNames[0];
    const output     = results[outputName].data;   // Float32Array

    // TODO: replace placeholder labels with real epitope region names
    // once the identification model output schema is confirmed.
    const ranked = Array.from(output)
      .map((score, i) => ({ value: `epitope-${i}`, label: `Epitope ${i + 1}`, score }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 10);

    populateIdentifyResults(ranked);

  } catch (err) {
    console.error('openMIP identification error:', err);
    showToast('Error: ' + err.message);
  } finally {
    btn.textContent = 'Identify';
    btn.classList.remove('running');
    btn.disabled = false;
  }
}

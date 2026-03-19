/* ── openMIP ONNX prediction pipeline ── */
// Depends on: onnxruntime-web (ort global), featurizePDB (below),
//             rawPdbText (viewer state, set by viewer.html).

let ortSession = null;   // cached InferenceSession

// Load (or return the cached) ONNX session from src/model.onnx.
async function loadModel() {
  if (ortSession) return ortSession;
  ortSession = await ort.InferenceSession.create('src/model.onnx');
  return ortSession;
}

// ── Featurizer ───────────────────────────────────────────────────────────────
// PLACEHOLDER — replace with the real feature extraction logic
// once the model input schema is finalised.
//
// Currently extracts a flat Float32Array of:
//   [atomCount, heavyAtomCount, C_count, H_count, N_count, O_count, S_count,
//    mean_x, mean_y, mean_z]
// from the raw PDB text.
function featurizePDB(pdbText) {
  const elemCount = {};
  let sumX = 0, sumY = 0, sumZ = 0, n = 0;

  for (const line of pdbText.split('\n')) {
    const rec = line.substring(0, 6).trim();
    if (rec !== 'ATOM' && rec !== 'HETATM') continue;
    const elemCol  = line.length > 76 ? line.substring(76, 78).trim() : '';
    const atomName = line.substring(12, 16).trim();
    const el = (elemCol || atomName.replace(/[^A-Za-z]/g, '')[0] || 'C').toUpperCase();
    elemCount[el] = (elemCount[el] || 0) + 1;
    sumX += parseFloat(line.substring(30, 38)) || 0;
    sumY += parseFloat(line.substring(38, 46)) || 0;
    sumZ += parseFloat(line.substring(46, 54)) || 0;
    n++;
  }

  const heavy = Object.entries(elemCount)
    .filter(([el]) => el !== 'H' && el !== 'D')
    .reduce((s, [, c]) => s + c, 0);

  return new Float32Array([
    n,
    heavy,
    elemCount['C'] || 0,
    elemCount['H'] || 0,
    elemCount['N'] || 0,
    elemCount['O'] || 0,
    elemCount['S'] || 0,
    n > 0 ? sumX / n : 0,
    n > 0 ? sumY / n : 0,
    n > 0 ? sumZ / n : 0,
  ]);
}

// Render a ranked list of { label, score } into #predict-results.
function showPredictResults(ranked) {
  const el = document.getElementById('predict-results');
  el.innerHTML = '';
  ranked.forEach(({ label, score }) => {
    const row = document.createElement('div');
    row.className = 'predict-row';
    row.innerHTML =
      `<span class="predict-row-label">${label}</span>` +
      `<span class="predict-row-score">${score}</span>`;
    el.appendChild(row);
  });
  el.classList.add('visible');
}

// Run the full prediction pipeline for the currently loaded molecule.
// Reads rawPdbText from the viewer scope.
async function runPrediction() {
  if (!rawPdbText) {
    showPredictResults([{ label: 'No molecule loaded', score: '—' }]);
    return;
  }

  const aaCount = countAAResidues(rawPdbText);
  if (aaCount >= 25) {
    showToast('Prediction is only supported for small molecules and short peptides.');
    return;
  }

  const btn = document.getElementById('predict');
  btn.textContent = 'Running…';
  btn.classList.add('running');
  btn.disabled = true;

  try {
    const session  = await loadModel();
    const features = featurizePDB(rawPdbText);

    // Build input tensor — name must match the model's expected input name.
    // Update the shape [1, features.length] if needed.
    const inputName   = session.inputNames[0];
    const inputTensor = new ort.Tensor('float32', features, [1, features.length]);
    const results     = await session.run({ [inputName]: inputTensor });

    // Interpret output — assumes a 1-D float32 output ranked by score.
    // Update the label list to match your model's output schema.
    const outputName = session.outputNames[0];
    const output     = results[outputName].data;   // Float32Array

    // TODO: replace these placeholder labels with real candidate names
    // once the model output schema is confirmed.
    const LABELS = Array.from({ length: output.length }, (_, i) => `Candidate ${i + 1}`);

    const ranked = Array.from(output)
      .map((score, i) => ({ label: LABELS[i] || `Output ${i}`, score }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 10)
      .map(({ label, score }) => ({ label, score: score.toFixed(4) }));

    showPredictResults(ranked);

  } catch (err) {
    console.error('openMIP prediction error:', err);
    showPredictResults([{ label: 'Error: ' + err.message, score: '—' }]);
  } finally {
    btn.textContent = 'Predict';
    btn.classList.remove('running');
    btn.disabled = false;
  }
}

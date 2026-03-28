/**
 * diffMIP Backend Client
 * Connects to Python Flask backend for inference (avoids ONNX export issues)
 *
 * This is simpler and faster than browser-based ONNX.js:
 * - No model loading in browser
 * - Faster inference (GPU support)
 * - Avoids E3NN → ONNX conversion problems
 *
 * Depends on: jQuery, backend_api.py running on localhost:5000
 */

const DIFFMIP_API_URL = 'http://localhost:5000';

/**
 * Check if backend is available
 */
async function checkDiffMIPBackend() {
  try {
    const response = await fetch(`${DIFFMIP_API_URL}/health`);
    if (!response.ok) return false;

    const data = await response.json();
    return data.status === 'ok' && data.models_loaded;

  } catch (e) {
    console.log('diffMIP backend not available:', e.message);
    return false;
  }
}

/**
 * Initialize diffMIP (check backend availability)
 */
async function initDiffMIP() {
  const statusEl = $('#diffmip-status');
  const btnEl = $('#diffmip-predict');

  statusEl.text('Checking backend...').css('color', 'var(--text-muted)');

  const available = await checkDiffMIPBackend();

  if (available) {
    statusEl.text('Ready (Backend)').css('color', '#4ade80');
    btnEl.prop('disabled', false);
    console.log('✓ diffMIP backend ready');
  } else {
    statusEl.html(
      'Backend offline<br>' +
      '<span style="font-size: 0.7rem; opacity: 0.7;">Start with: python backend_api.py</span>'
    ).css('color', '#f87171');
    btnEl.prop('disabled', true);
    console.warn('diffMIP backend not running');
  }
}

/**
 * Run diffMIP prediction via backend API
 */
async function runDiffMIP() {
  if (!rawPdbText) {
    showToast('Please load a molecule first');
    return;
  }

  const btn = $('#diffmip-predict');
  const statusEl = $('#diffmip-status');

  btn.text('Running…').addClass('running').prop('disabled', true);
  statusEl.text('Running inference...').css('color', '#fbbf24');

  try {
    const numMonomers = parseInt($('#diffmip-num-monomers').val()) || 5;
    const numSamples = parseInt($('#diffmip-num-samples').val()) || 1;
    const numSteps = parseInt($('#diffmip-num-steps').val()) || 50;

    console.log(`diffMIP inference: ${numMonomers} monomers, ${numSamples} samples, ${numSteps} steps`);

    // Call backend API
    const response = await fetch(`${DIFFMIP_API_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        pdb: rawPdbText,
        num_monomers: numMonomers,
        num_samples: numSamples,
        num_steps: numSteps
      })
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: response.statusText }));
      throw new Error(errorData.error || `API error: ${response.statusText}`);
    }

    const results = await response.json();
    console.log('✓ Received results:', results);

    // Show results
    showDiffMIPResults(results);
    showToast(`Generated ${results.placements.length} monomer placements`);

    statusEl.text('Ready (Backend)').css('color', '#4ade80');

  } catch (error) {
    console.error('diffMIP error:', error);
    showToast('Error: ' + error.message);
    statusEl.text('Error - see console').css('color', '#f87171');
  } finally {
    btn.text('Generate Placements').removeClass('running').prop('disabled', false);
  }
}

/**
 * Display diffMIP results
 */
function showDiffMIPResults(results) {
  const container = $('#diffmip-results');
  container.empty();

  if (!results.placements || results.placements.length === 0) {
    container.html('<div style="color: var(--text-muted); font-size: 0.85rem;">No placements generated</div>');
    container.show();
    return;
  }

  // Group by sample
  const bySample = {};
  results.placements.forEach(p => {
    if (!bySample[p.sample]) bySample[p.sample] = [];
    bySample[p.sample].push(p);
  });

  // Create sample sections
  Object.entries(bySample).forEach(([sampleNum, placements]) => {
    const sampleDiv = $('<div class="diffmip-sample">');
    sampleDiv.append(`<div class="diffmip-sample-header">Sample ${sampleNum}</div>`);

    placements.forEach(p => {
      const row = $('<div class="diffmip-placement-row">');
      row.html(`
        <span class="diffmip-monomer-code">${p.code}</span>
        <span class="diffmip-monomer-name">${p.fullname || p.code}</span>
        <span class="diffmip-centroid">(${p.centroid.map(v => v.toFixed(1)).join(', ')})</span>
      `);

      // Click to visualize (optional - implement visualizePlacement)
      row.on('click', function() {
        visualizePlacement(p);
        $(this).addClass('selected').siblings().removeClass('selected');
      });

      sampleDiv.append(row);
    });

    container.append(sampleDiv);
  });

  container.show();
}

/**
 * Visualize a placement in 3Dmol viewer (optional enhancement)
 */
function visualizePlacement(placement) {
  // Optional: Add sphere at centroid position
  if (typeof viewer !== 'undefined') {
    // Clear previous markers
    viewer.removeAllShapes();

    // Add sphere at placement centroid
    viewer.addSphere({
      center: {
        x: placement.centroid[0],
        y: placement.centroid[1],
        z: placement.centroid[2]
      },
      radius: 1.0,
      color: '#4ade80',
      alpha: 0.7
    });

    viewer.render();
    console.log(`Visualized placement: ${placement.code} at (${placement.centroid.join(', ')})`);
  }
}

/**
 * Clear placement visualizations
 */
function clearPlacementVisualizations() {
  if (typeof viewer !== 'undefined') {
    viewer.removeAllShapes();
    viewer.render();
  }
}

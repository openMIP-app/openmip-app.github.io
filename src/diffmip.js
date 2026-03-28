/**
 * diffMIP Web Inference
 * Browser-based monomer placement using ONNX.js and flow matching
 *
 * This module implements flow matching inference in pure JavaScript:
 * - Loads three ONNX models (encoder, centroid_net, torsion_net)
 * - Implements Euler integration for flow matching
 * - Generates monomer placements around target molecules
 *
 * Depends on: onnxruntime-web (ort global), 3Dmol.js, jQuery
 */

class DiffMIPWeb {
  constructor() {
    this.encoderSession = null;
    this.centroidSession = null;
    this.torsionSession = null;
    this.config = null;
    this.library = null;
    this.isLoaded = false;
  }

  /**
   * Load all ONNX models and configuration
   * @param {string} modelDir - Directory containing ONNX models
   */
  async loadModels(modelDir = 'models/diffmip') {
    try {
      console.log('Loading diffMIP models...');

      // Load configuration
      const configResponse = await fetch(`${modelDir}/config.json`);
      this.config = await configResponse.json();
      console.log('  ✓ Loaded config');

      // Load monomer library
      const libraryResponse = await fetch(`${modelDir}/fm-list.yaml`);
      const libraryText = await libraryResponse.text();
      this.library = this.parseLibrary(libraryText);
      console.log(`  ✓ Loaded library (${Object.keys(this.library).length} monomers)`);

      // Load ONNX models
      this.encoderSession = await ort.InferenceSession.create(`${modelDir}/target_encoder.onnx`);
      console.log('  ✓ Loaded target encoder');

      this.centroidSession = await ort.InferenceSession.create(`${modelDir}/centroid_score_net.onnx`);
      console.log('  ✓ Loaded centroid network');

      this.torsionSession = await ort.InferenceSession.create(`${modelDir}/torsion_score_net.onnx`);
      console.log('  ✓ Loaded torsion network');

      this.isLoaded = true;
      console.log('✓ diffMIP models loaded successfully');
      return true;

    } catch (error) {
      console.error('Failed to load diffMIP models:', error);
      throw new Error(`Model loading failed: ${error.message}`);
    }
  }

  /**
   * Parse YAML library file (simplified parser)
   */
  parseLibrary(yamlText) {
    const library = {};
    const lines = yamlText.split('\n');
    let currentCode = null;
    let inFmList = false;

    for (const line of lines) {
      if (line.trim() === 'fm-list:') {
        inFmList = true;
        continue;
      }

      if (inFmList) {
        // Match monomer code (e.g., "  AAM:")
        const codeMatch = line.match(/^  ([A-Z0-9-]+):/);
        if (codeMatch) {
          currentCode = codeMatch[1];
          library[currentCode] = { code: currentCode };
          continue;
        }

        // Match properties
        if (currentCode) {
          const smilesMatch = line.match(/smiles:\s*"?([^"]+)"?/);
          const fullnameMatch = line.match(/fullname:\s*"([^"]+)"/);

          if (smilesMatch) library[currentCode].smiles = smilesMatch[1];
          if (fullnameMatch) library[currentCode].fullname = fullnameMatch[1];
        }
      }
    }

    return library;
  }

  /**
   * Parse PDB text and extract target structure
   * @param {string} pdbText - PDB format text
   * @returns {Object} Target structure with pos, atomic_nums, elements, x, edge_index
   */
  parsePDB(pdbText) {
    const atoms = [];

    // Parse ATOM/HETATM records
    for (const line of pdbText.split('\n')) {
      const record = line.substring(0, 6).trim();
      if (record !== 'ATOM' && record !== 'HETATM') continue;

      const element = line.length > 76 ? line.substring(76, 78).trim() :
                      line.substring(12, 16).trim().replace(/[^A-Za-z]/g, '')[0] || 'C';
      const x = parseFloat(line.substring(30, 38));
      const y = parseFloat(line.substring(38, 46));
      const z = parseFloat(line.substring(46, 54));

      // Map element to atomic number
      const atomicNumMap = {
        'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9,
        'P': 15, 'S': 16, 'Cl': 17, 'Br': 35
      };
      const atomicNum = atomicNumMap[element] || 6;

      atoms.push({ x, y, z, element, atomicNum });
    }

    if (atoms.length === 0) {
      throw new Error('No atoms found in PDB');
    }

    // Center at origin
    const meanX = atoms.reduce((s, a) => s + a.x, 0) / atoms.length;
    const meanY = atoms.reduce((s, a) => s + a.y, 0) / atoms.length;
    const meanZ = atoms.reduce((s, a) => s + a.z, 0) / atoms.length;

    atoms.forEach(a => {
      a.x -= meanX;
      a.y -= meanY;
      a.z -= meanZ;
    });

    // Build structure
    const N = atoms.length;
    const pos = new Float32Array(N * 3);
    const atomic_nums = new Int32Array(N);
    const elements = [];

    atoms.forEach((atom, i) => {
      pos[i * 3] = atom.x;
      pos[i * 3 + 1] = atom.y;
      pos[i * 3 + 2] = atom.z;
      atomic_nums[i] = atom.atomicNum;
      elements.push(atom.element);
    });

    // Build one-hot features (9 atom types)
    const ATOM_TYPES = [1, 6, 7, 8, 9, 15, 16, 17, 35];
    const x = new Float32Array(N * ATOM_TYPES.length);
    for (let i = 0; i < N; i++) {
      const idx = ATOM_TYPES.indexOf(atomic_nums[i]);
      if (idx >= 0) {
        x[i * ATOM_TYPES.length + idx] = 1.0;
      }
    }

    // Build edges (radius-based)
    const maxRadius = this.config.max_radius || 6.0;
    const edges = [];
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        if (i === j) continue;
        const dx = pos[i*3] - pos[j*3];
        const dy = pos[i*3+1] - pos[j*3+1];
        const dz = pos[i*3+2] - pos[j*3+2];
        const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
        if (dist < maxRadius) {
          edges.push(i, j);
        }
      }
    }

    const edge_index = new BigInt64Array(edges);

    return {
      pos,           // (N, 3) positions
      atomic_nums,   // (N,) atomic numbers
      elements,      // (N,) element symbols
      x,             // (N, 9) one-hot features
      edge_index,    // (2, E) COO edges
      N,             // number of atoms
      E: edges.length / 2  // number of edges
    };
  }

  /**
   * Encode target molecule
   */
  async encodeTarget(target) {
    const batch = new BigInt64Array(target.N).fill(0n);

    const feeds = {
      x: new ort.Tensor('float32', target.x, [target.N, 9]),
      pos: new ort.Tensor('float32', target.pos, [target.N, 3]),
      edge_index: new ort.Tensor('int64', target.edge_index, [2, target.E]),
      batch: new ort.Tensor('int64', batch, [target.N])
    };

    const results = await this.encoderSession.run(feeds);
    return {
      node_emb: results.node_emb,
      global_ctx: results.global_ctx
    };
  }

  /**
   * Sample centroid positions using flow matching
   */
  async sampleCentroids(numMonomers, monomerTypes, targetEmbeddings, targetBatch, numSteps = 50) {
    const B = 1;
    const K = numMonomers;
    const T = 1000; // Training timestep convention

    // Start from Gaussian noise
    const x = new Float32Array(B * K * 3);
    for (let i = 0; i < x.length; i++) {
      x[i] = this.randomNormal();
    }

    const dt = 1.0 / numSteps;
    const mono_type_idx = new BigInt64Array(B * K);
    for (let i = 0; i < K; i++) {
      mono_type_idx[i] = BigInt(monomerTypes[i]);
    }

    // Euler integration
    for (let step = 0; step < numSteps; step++) {
      const t = step / numSteps;
      const t_scaled = BigInt(Math.floor(t * (T - 1)));
      const t_array = new BigInt64Array(B).fill(t_scaled);

      // Predict velocity
      const feeds = {
        centroid_t: new ort.Tensor('float32', x, [B, K, 3]),
        mono_type_idx: new ort.Tensor('int64', mono_type_idx, [B, K]),
        t: new ort.Tensor('int64', t_array, [B]),
        target_node_emb: targetEmbeddings.node_emb,
        target_global_ctx: targetEmbeddings.global_ctx,
        target_batch: new ort.Tensor('int64', targetBatch, [targetBatch.length])
      };

      const result = await this.centroidSession.run(feeds);
      const velocity = result.velocity.data;

      // Euler step: x = x + dt * velocity
      for (let i = 0; i < x.length; i++) {
        x[i] += dt * velocity[i];
      }
    }

    return x; // (B, K, 3) flattened
  }

  /**
   * Sample torsion DOFs using flow matching
   */
  async sampleTorsions(numMonomers, monomerTypes, centroids, targetEmbeddings, targetBatch, numSteps = 50) {
    const B = 1;
    const K = numMonomers;
    const D_dof = 3 + this.config.max_torsions_per_monomer;

    // Start from uniform on torus
    const dof = new Float32Array(B * K * D_dof);
    for (let i = 0; i < dof.length; i++) {
      dof[i] = Math.random() * 2 * Math.PI - Math.PI;
    }

    // DOF mask (all enabled)
    const dof_mask = new Uint8Array(B * K * D_dof).fill(1);

    const dt = 1.0 / numSteps;
    const sigma_max = this.config.sigma_max_torsion || Math.PI;
    const sigma_min = this.config.sigma_min_torsion || 0.01 * Math.PI;

    const mono_type_idx = new BigInt64Array(B * K);
    for (let i = 0; i < K; i++) {
      mono_type_idx[i] = BigInt(monomerTypes[i]);
    }

    // Euler integration on torus
    for (let step = 0; step < numSteps; step++) {
      const t = step / numSteps;

      // Map t to sigma (reverse: t=0 → high, t=1 → low)
      const sigma_val = sigma_max * (1 - t) + sigma_min * t;
      const sigma = new Float32Array(B).fill(sigma_val);

      // Predict velocity
      const feeds = {
        dof_t: new ort.Tensor('float32', dof, [B, K, D_dof]),
        dof_mask: new ort.Tensor('bool', dof_mask, [B, K, D_dof]),
        centroid_0: new ort.Tensor('float32', centroids, [B, K, 3]),
        mono_type_idx: new ort.Tensor('int64', mono_type_idx, [B, K]),
        sigma: new ort.Tensor('float32', sigma, [B]),
        target_node_emb: targetEmbeddings.node_emb,
        target_global_ctx: targetEmbeddings.global_ctx,
        target_batch: new ort.Tensor('int64', targetBatch, [targetBatch.length])
      };

      const result = await this.torsionSession.run(feeds);
      const velocity = result.velocity.data;

      // Euler step on torus
      for (let i = 0; i < dof.length; i++) {
        dof[i] += dt * velocity[i];
        // Wrap to [-π, π]
        dof[i] = Math.atan2(Math.sin(dof[i]), Math.cos(dof[i]));
      }
    }

    return dof; // (B, K, D_dof) flattened
  }

  /**
   * Run full diffMIP inference
   * @param {string} pdbText - PDB text of target molecule
   * @param {number} numMonomers - Number of monomers to place
   * @param {number} numSamples - Number of independent samples
   * @param {number} numSteps - Number of integration steps
   */
  async predict(pdbText, numMonomers = 5, numSamples = 1, numSteps = 25) {
    if (!this.isLoaded) {
      throw new Error('Models not loaded. Call loadModels() first.');
    }

    console.log(`Running diffMIP inference...`);
    console.log(`  Monomers: ${numMonomers}`);
    console.log(`  Samples: ${numSamples}`);
    console.log(`  Steps: ${numSteps}`);

    // Parse target
    const target = this.parsePDB(pdbText);
    console.log(`  Target atoms: ${target.N}`);

    // Encode target
    const targetBatch = new BigInt64Array(target.N).fill(0n);
    const embeddings = await this.encodeTarget(target);
    console.log(`  ✓ Encoded target`);

    // Select random monomer types
    const librarySize = this.config.num_monomer_types;
    const placements = [];

    for (let sampleIdx = 0; sampleIdx < numSamples; sampleIdx++) {
      // Random monomer selection
      const monomerTypes = [];
      for (let i = 0; i < numMonomers; i++) {
        monomerTypes.push(Math.floor(Math.random() * librarySize));
      }

      // Sample centroids
      const centroids = await this.sampleCentroids(
        numMonomers, monomerTypes, embeddings, targetBatch, numSteps
      );
      console.log(`  ✓ Sample ${sampleIdx + 1}: centroids generated`);

      // Sample torsions
      const dofs = await this.sampleTorsions(
        numMonomers, monomerTypes, centroids, embeddings, targetBatch, numSteps
      );
      console.log(`  ✓ Sample ${sampleIdx + 1}: torsions generated`);

      // Extract results
      const D_dof = 3 + this.config.max_torsions_per_monomer;
      const libraryCodes = Object.keys(this.library);

      for (let k = 0; k < numMonomers; k++) {
        const monomerIdx = monomerTypes[k];
        const code = monomerIdx < libraryCodes.length ? libraryCodes[monomerIdx] : `MON${monomerIdx}`;
        const mono = this.library[code] || { code, fullname: code };

        placements.push({
          sample: sampleIdx + 1,
          k: k + 1,
          code: mono.code,
          fullname: mono.fullname || mono.code,
          centroid: [
            centroids[k * 3],
            centroids[k * 3 + 1],
            centroids[k * 3 + 2]
          ],
          euler: [
            dofs[k * D_dof],
            dofs[k * D_dof + 1],
            dofs[k * D_dof + 2]
          ]
        });
      }
    }

    console.log(`✓ Generated ${placements.length} placements`);

    return {
      placements,
      target: {
        pos: Array.from(target.pos),
        elements: target.elements
      }
    };
  }

  /**
   * Generate random number from standard normal distribution
   */
  randomNormal() {
    // Box-Muller transform
    const u1 = Math.random();
    const u2 = Math.random();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }
}

// Global instance
const diffmipWeb = new DiffMIPWeb();

/**
 * Initialize diffMIP (called on page load)
 */
async function initDiffMIP() {
  try {
    await diffmipWeb.loadModels('models/diffmip');
    $('#diffmip-status').text('Ready').css('color', '#4ade80');
    $('#diffmip-predict').prop('disabled', false);
  } catch (error) {
    console.error('Failed to initialize diffMIP:', error);
    $('#diffmip-status').text('Error: ' + error.message).css('color', '#f87171');
  }
}

/**
 * Run diffMIP prediction (called by button click)
 */
async function runDiffMIP() {
  if (!rawPdbText) {
    showToast('Please load a molecule first');
    return;
  }

  const btn = $('#diffmip-predict');
  btn.text('Running…').addClass('running').prop('disabled', true);

  try {
    const numMonomers = parseInt($('#diffmip-num-monomers').val()) || 5;
    const numSamples = parseInt($('#diffmip-num-samples').val()) || 1;
    const numSteps = parseInt($('#diffmip-num-steps').val()) || 25;

    const results = await diffmipWeb.predict(rawPdbText, numMonomers, numSamples, numSteps);

    // Show results
    showDiffMIPResults(results);
    showToast(`Generated ${results.placements.length} monomer placements`);

  } catch (error) {
    console.error('diffMIP error:', error);
    showToast('Error: ' + error.message);
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
    return;
  }

  // Group by sample
  const bySample = {};
  results.placements.forEach(p => {
    if (!bySample[p.sample]) bySample[p.sample] = [];
    bySample[p.sample].push(p);
  });

  Object.entries(bySample).forEach(([sampleNum, placements]) => {
    const sampleDiv = $('<div class="diffmip-sample">');
    sampleDiv.append(`<div class="diffmip-sample-header">Sample ${sampleNum}</div>`);

    placements.forEach(p => {
      const row = $('<div class="diffmip-placement-row">');
      row.html(`
        <span class="diffmip-monomer-code">${p.code}</span>
        <span class="diffmip-monomer-name">${p.fullname}</span>
        <span class="diffmip-centroid">(${p.centroid.map(v => v.toFixed(1)).join(', ')})</span>
      `);
      sampleDiv.append(row);
    });

    container.append(sampleDiv);
  });

  container.show();
}

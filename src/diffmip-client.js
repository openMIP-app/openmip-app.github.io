/**
 * diffMIP Client-Side Inference
 *
 * Complete browser-based inference using ONNX Runtime Web
 * No backend server required - all computation runs in the browser
 *
 * Dependencies:
 * - onnxruntime-web (https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js)
 * - js-yaml (for monomer library parsing)
 */

// Global state
let diffmipModels = null;
let monomerLibrary = null;
let rdkitModule = null;  // RDKit.js module for client-side structure generation

// Stub functions - progress modal removed
function updateProgress(stage, details, progress = null) {
  // No-op: progress modal removed
}

function clearProgressLog() {
  // No-op: progress modal removed
}

// Feature flags
const ENABLE_MONOMER_MONOMER_INTERACTIONS = false;  // Set to true to penalize overlapping monomers

// TEMPORARY: Disable torsional application for debugging
const DISABLE_TORSION_APPLICATION = false;  // Set to true to skip torsional rotations

// Recipe mode: iterative monomer placement (like diffmip recipe.py)
const USE_RECIPE_MODE = true;  // Set to true for iterative placement, false for single-shot

// IMPORTANT: Model capacity - embedding layer supports max 107 monomer types
// Even if library has more, we must limit to what model supports
const MAX_MODEL_TYPES = 107;

// Configuration
const DIFFMIP_CONFIG = {
  modelPath: 'models/',
  encoderModel: 'encoder_quantized.onnx',
  centroidModel: 'centroid_score_quantized.onnx',
  torsionModel: 'torsion_score_quantized.onnx',
  screeningModel: 'screening_quantized.onnx',
  libraryPath: 'models/fm-list.yaml',

  // Model architecture parameters (must match trained model)
  nodeFeatDim: 24,
  hiddenDim: 256,
  contextDim: 256,
  maxDof: 20,
  maxRadius: 10.0,
};

/**
 * Generate a standard normal random number (Gaussian with mean=0, std=1)
 * Uses Box-Muller transform
 */
function randn() {
  // Box-Muller transform
  let u1 = 0, u2 = 0;
  while (u1 === 0) u1 = Math.random(); // Converting [0,1) to (0,1)
  while (u2 === 0) u2 = Math.random();

  const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  return z0;
}

/**
 * Wait for dependencies to load
 */
async function waitForDependencies(maxWaitMs = 10000) {
  const startTime = Date.now();

  while (Date.now() - startTime < maxWaitMs) {
    if (typeof ort !== 'undefined' &&
        typeof jsyaml !== 'undefined' &&
        typeof $ !== 'undefined' &&
        typeof window.initRDKitModule !== 'undefined') {
      return true;
    }
    await new Promise(resolve => setTimeout(resolve, 100));
  }

  return false;
}

/**
 * Initialize RDKit.js module
 */
async function initRDKit() {
  if (rdkitModule) return rdkitModule;

  try {
    console.log('Initializing RDKit.js...');
    rdkitModule = await window.initRDKitModule();
    console.log('✓ RDKit.js initialized');
    return rdkitModule;
  } catch (error) {
    console.error('Failed to initialize RDKit.js:', error);
    return null;
  }
}


/**
 * Initialize ONNX Runtime and load models
 */
async function initDiffMIPClient() {
  // Wait for dependencies to load
  console.log('Waiting for dependencies (ONNX Runtime, js-yaml, jQuery)...');
  const depsReady = await waitForDependencies();

  if (!depsReady) {
    console.error('Dependencies failed to load after 5 seconds');
    return false;
  }

  console.log('✓ Dependencies loaded');
  console.log('='.repeat(70));
  console.log('Loading diffMIP-lite client-side models...');
  console.log('='.repeat(70));

  const statusEl = $('#diffmip-status');
  const btnEl = $('#diffmip-predict');

  // Check if elements exist
  if (!statusEl.length || !btnEl.length) {
    console.error('diffMIP UI elements not found in DOM');
    return false;
  }

  statusEl.text('Loading models...').css('color', 'var(--text-muted)');
  btnEl.prop('disabled', true);

  try {
    // Check if ONNX Runtime is available
    if (typeof ort === 'undefined') {
      throw new Error('ONNX Runtime not loaded. Include onnxruntime-web script in <head>');
    }
    console.log('✓ ONNX Runtime available');

    // Check if js-yaml is available
    if (typeof jsyaml === 'undefined') {
      throw new Error('js-yaml not loaded. Include js-yaml script in <head>');
    }
    console.log('✓ js-yaml available');

    // Configure ONNX Runtime with GPU acceleration if available
    // Execution provider priority: WebGPU (fastest) > WebGL > WASM (fallback)
    const sessionOptions = {
      executionProviders: [
        'webgpu',  // Try WebGPU first (Chrome 113+, Edge 113+)
        'webgl',   // Fall back to WebGL (widely supported)
        'wasm'     // Final fallback (CPU)
      ],
      graphOptimizationLevel: 'all',
      enableCpuMemArena: false,  // Disable arena to reduce memory
      enableMemPattern: false,    // Disable pattern to reduce memory
      executionMode: 'sequential',
      logSeverityLevel: 3,        // Only errors
      logVerbosityLevel: 0,       // Minimal logging
    };

    // Set WASM memory options (before loading models)
    if (typeof ort !== 'undefined' && ort.env) {
      ort.env.wasm.numThreads = 1;  // Single thread = less memory
      ort.env.wasm.simd = true;      // Keep SIMD for performance

      // Increase WASM memory limits to reduce allocation warnings
      // Note: These are hints to the browser, not guarantees
      try {
        ort.env.wasm.proxy = false;  // Disable worker proxy to save memory
      } catch (e) {
        console.warn('[loadModels] Could not set WASM proxy option:', e.message);
      }
    }

    // Log WebAssembly memory warning advice
    console.log('%c[Memory Notice]', 'color: #fbbf24; font-weight: bold');
    console.log('If you see "failed to allocate executable memory" warnings:');
    console.log('  • This is normal for WASM JIT compilation (not a problem with our code)');
    console.log('  • Models are only 4.4 MB total and loaded once');
    console.log('  • Warning is about executable memory for JIT code, not model data');
    console.log('  • Models should still work correctly (warning is non-fatal)');
    console.log('  • For best performance: close unused tabs if memory is limited');

    // Load models SEQUENTIALLY to avoid memory issues
    // Loading in parallel can exhaust WASM memory
    console.log('\nLoading ONNX models sequentially...');

    statusEl.text('Loading encoder (1/3)...').css('color', '#fbbf24');
    console.log(`  Encoder: ${DIFFMIP_CONFIG.modelPath}${DIFFMIP_CONFIG.encoderModel}`);
    const encoder = await ort.InferenceSession.create(
      `${DIFFMIP_CONFIG.modelPath}${DIFFMIP_CONFIG.encoderModel}`,
      sessionOptions
    ).catch(e => {
      throw new Error(`Encoder load failed: ${e.message.substring(0, 200)}`);
    });
    console.log('  ✓ Encoder loaded');

    // Log which execution provider is actually being used
    try {
      const usedProviders = encoder.handler?.executionProviders ||
                           encoder.outputNames?.map(() => 'unknown') ||
                           ['unknown'];
      const providerName = Array.isArray(usedProviders) ? usedProviders[0] : usedProviders;
      console.log(`  → Using execution provider: ${providerName}`);

      if (providerName === 'webgpu') {
        console.log('  🚀 GPU acceleration enabled (WebGPU)');
      } else if (providerName === 'webgl') {
        console.log('  🎨 GPU acceleration enabled (WebGL)');
      } else if (providerName === 'wasm') {
        console.log('  💻 Using CPU (WASM) - GPU not available');
      }
    } catch (e) {
      console.log('  → Could not determine execution provider');
    }

    await new Promise(resolve => setTimeout(resolve, 200)); // GC pause (increased for memory cleanup)

    statusEl.text('Loading centroid model (2/3)...').css('color', '#fbbf24');
    console.log(`  Centroid: ${DIFFMIP_CONFIG.modelPath}${DIFFMIP_CONFIG.centroidModel}`);
    const centroidNet = await ort.InferenceSession.create(
      `${DIFFMIP_CONFIG.modelPath}${DIFFMIP_CONFIG.centroidModel}`,
      sessionOptions
    ).catch(e => {
      throw new Error(`Centroid model load failed: ${e.message.substring(0, 200)}`);
    });
    console.log('  ✓ Centroid model loaded');
    await new Promise(resolve => setTimeout(resolve, 200)); // GC pause (increased for memory cleanup)

    statusEl.text('Loading torsion model (3/3)...').css('color', '#fbbf24');
    console.log(`  Torsion: ${DIFFMIP_CONFIG.modelPath}${DIFFMIP_CONFIG.torsionModel}`);
    const torsionNet = await ort.InferenceSession.create(
      `${DIFFMIP_CONFIG.modelPath}${DIFFMIP_CONFIG.torsionModel}`,
      sessionOptions
    ).catch(e => {
      throw new Error(`Torsion model load failed: ${e.message.substring(0, 200)}`);
    });
    console.log('  ✓ Torsion model loaded');
    await new Promise(resolve => setTimeout(resolve, 200)); // GC pause (increased for memory cleanup)

    // Load screening model (SimpleScreeningNet - neural network approach)
    statusEl.text('Loading screening model...').css('color', '#fbbf24');
    let screeningNet = null;
    try {
      screeningNet = await ort.InferenceSession.create(
        `${DIFFMIP_CONFIG.modelPath}${DIFFMIP_CONFIG.screeningModel}`,
        sessionOptions
      );
      statusEl.text('Screening model loaded (neural)').css('color', '#22c55e');
      console.log('  ✓ Screening model loaded (neural network)');
    } catch (e) {
      statusEl.text('Screening: physics fallback').css('color', '#fbbf24');
      console.log('  ⚠ Screening model not found, will use physics-based fallback');
      console.log('    To use neural screening, ensure screening_quantized.onnx is in models/');
    }
    await new Promise(resolve => setTimeout(resolve, 200)); // GC pause (increased for memory cleanup)

    console.log('\n✓ All ONNX models loaded successfully');
    if (screeningNet) {
      console.log('✓ Screening: Using neural network (SimpleScreeningNet)');
    } else {
      console.log('✓ Screening: Using physics-based fallback');
    }

    // Load monomer library
    console.log(`\nLoading monomer library: ${DIFFMIP_CONFIG.libraryPath}`);
    const libraryResponse = await fetch(DIFFMIP_CONFIG.libraryPath);
    if (!libraryResponse.ok) {
      throw new Error(`Failed to fetch library: ${libraryResponse.status} ${libraryResponse.statusText}`);
    }
    const libraryYaml = await libraryResponse.text();
    const libraryData = jsyaml.load(libraryYaml);

    // The YAML structure is: { "fm-list": { "AAC": {...}, "AAM": {...}, ... } }
    // Convert to array format: [{ code: "AAC", ...data }, { code: "AAM", ...data }, ...]
    const libraryDict = libraryData['fm-list'];
    if (!libraryDict || typeof libraryDict !== 'object') {
      throw new Error('Invalid library format: expected "fm-list" key with monomer dictionary');
    }

    const library = Object.entries(libraryDict).map(([code, data]) => ({
      code: code,
      ...data
    }));

    console.log(`✓ Parsed ${library.length} monomers from YAML`);

    // Build lookup tables
    monomerLibrary = {
      monomers: library,
      numTypes: library.length,
      codeToIdx: {},
      idxToCode: {}
    };

    library.forEach((mono, idx) => {
      monomerLibrary.codeToIdx[mono.code] = idx;
      monomerLibrary.idxToCode[idx] = mono.code;
    });

    console.log(`✓ Loaded ${monomerLibrary.numTypes} monomers`);

    // Initialize RDKit.js for client-side structure generation
    statusEl.text('Loading RDKit.js...').css('color', '#fbbf24');
    await initRDKit();

    // NOW compute torsion_quads for each monomer (after RDKit is initialized)
    statusEl.text('Computing rotatable bonds...').css('color', '#fbbf24');
    console.log('Computing rotatable bonds for monomers...');
    for (let i = 0; i < Math.min(library.length, MAX_MODEL_TYPES); i++) {
      const mono = library[i];
      if (mono.smiles) {
        try {
          const structure = generateStructureFromSMILES(mono.smiles);
          if (structure && structure.torsionQuads) {
            mono.torsion_quads = structure.torsionQuads;
            if (i < 5) {  // Log first few for debugging
              console.log(`  ${mono.code}: ${mono.torsion_quads.length} rotatable bonds`);
            }
          }
        } catch (e) {
          console.warn(`  Failed to compute torsions for ${mono.code}:`, e.message);
        }
      }
    }
    console.log(`✓ Computed rotatable bonds for ${library.filter(m => m.torsion_quads).length}/${Math.min(library.length, MAX_MODEL_TYPES)} monomers`);

    // Store models
    diffmipModels = {
      encoder,
      centroidNet,
      torsionNet,
      screeningNet
    };

    console.log('='.repeat(70));
    console.log('✓ Client-side models ready!');
    console.log('='.repeat(70));

    statusEl.text('Ready (Client-Side)').css('color', '#4ade80');
    btnEl.prop('disabled', false);

    // Also enable the main Predict button
    $('#predict').prop('disabled', false);

    return true;

  } catch (error) {
    console.error('Failed to load diffMIP models:', error);
    console.error('Error details:', error.message);
    console.error('Stack:', error.stack);

    // Check if it's a memory error
    const isMemoryError = error.message.includes('out of memory') ||
                         error.message.includes('Aborted');

    // Show user-friendly error
    if (isMemoryError) {
      statusEl.html(
        `Out of Memory<br>` +
        `<span style="font-size: 0.65rem; opacity: 0.8;">Models too large for browser.<br>` +
        `Try: Refresh page, close tabs, or use desktop browser.</span>`
      ).css('color', '#f87171');

      console.error('='.repeat(70));
      console.error('MEMORY ERROR: ONNX models too large for WebAssembly');
      console.error('='.repeat(70));
      console.error('Solutions:');
      console.error('1. Refresh the page and try again');
      console.error('2. Close other browser tabs');
      console.error('3. Use Chrome/Edge (better WASM support)');
      console.error('4. Use a desktop computer (more RAM)');
      console.error('5. Quantize models: python quantize_models.py');
      console.error('='.repeat(70));
    } else {
      statusEl.html(
        `Failed to load<br>` +
        `<span style="font-size: 0.65rem; opacity: 0.8;">${error.message.substring(0, 100)}</span>`
      ).css('color', '#f87171');
    }

    btnEl.prop('disabled', true);
    $('#predict').prop('disabled', true);

    return false;
  }
}


/**
 * Parse PDB text to extract atoms
 */
function parsePDB(pdbText) {
  const atoms = [];
  const lines = pdbText.split('\n');

  for (const line of lines) {
    if (!line.startsWith('ATOM') && !line.startsWith('HETATM')) continue;

    // Parse PDB format
    const element = line.substring(76, 78).trim() || line.substring(12, 16).trim()[0];
    const x = parseFloat(line.substring(30, 38));
    const y = parseFloat(line.substring(38, 46));
    const z = parseFloat(line.substring(46, 54));

    if (isNaN(x) || isNaN(y) || isNaN(z)) continue;

    atoms.push({ element, x, y, z });
  }

  return atoms;
}


/**
 * Get atomic number from element symbol
 */
function getAtomicNumber(element) {
  const atomicNumbers = {
    'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 'S': 16, 'CL': 17, 'BR': 35, 'I': 53
  };
  return atomicNumbers[element.toUpperCase()] || 6; // Default to carbon
}


/**
 * Convert atomic numbers to one-hot features (dim=24)
 */
function atomicNumsToFeat(atomicNums) {
  const N = atomicNums.length;
  const features = new Float32Array(N * 24);

  // Simple one-hot encoding for common atoms
  // [H, C, N, O, F, P, S, Cl, Br, I, ...others]
  const atomToIdx = { 1: 0, 6: 1, 7: 2, 8: 3, 9: 4, 15: 5, 16: 6, 17: 7, 35: 8, 53: 9 };

  for (let i = 0; i < N; i++) {
    const atomNum = atomicNums[i];
    const featIdx = atomToIdx[atomNum] !== undefined ? atomToIdx[atomNum] : 10;
    if (featIdx < 24) {
      features[i * 24 + featIdx] = 1.0;
    }
  }

  return features;
}


/**
 * Build edges based on distance threshold (radius graph)
 * DEPRECATED: Use buildBondEdgeIndex() for chemical bond-based edges instead
 * This is kept for backward compatibility but causes train/test mismatch
 */
function buildRadiusEdges(positions, radius) {
  const N = positions.length / 3;
  const edges = [];

  for (let i = 0; i < N; i++) {
    for (let j = i + 1; j < N; j++) {
      const dx = positions[i * 3 + 0] - positions[j * 3 + 0];
      const dy = positions[i * 3 + 1] - positions[j * 3 + 1];
      const dz = positions[i * 3 + 2] - positions[j * 3 + 2];
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

      if (dist <= radius) {
        edges.push(i, j);
        edges.push(j, i); // Undirected edge
      }
    }
  }

  return new BigInt64Array(edges.map(e => BigInt(e)));
}

/**
 * Build bond-based edge index from chemical bond topology
 * This matches the training pipeline which uses mol_to_graph() from RDKit
 *
 * @param {Array} bonds - Bond array from RDKit JSON: [{atoms: [i,j], bo: bondOrder}, ...]
 * @returns {BigInt64Array} - Edge index as [2, E] flattened COO format for int64 tensors
 */
function buildBondEdgeIndex(bonds) {
  const edges = [];

  // Convert bonds to bidirectional edges (same as training)
  for (const bond of bonds) {
    const [i, j] = bond.atoms;

    // Add both directions (i->j and j->i) for undirected graph
    edges.push([i, j]);
    edges.push([j, i]);
  }

  // Convert to [2, E] COO format: [sources..., targets...]
  // Use BigInt64Array for ONNX int64 tensors
  const edgeIndex = new BigInt64Array(edges.length * 2);
  for (let e = 0; e < edges.length; e++) {
    edgeIndex[e] = BigInt(edges[e][0]);                // Source indices (first half)
    edgeIndex[edges.length + e] = BigInt(edges[e][1]); // Target indices (second half)
  }

  return edgeIndex;
}

/**
 * Infer chemical bonds from atomic coordinates and elements
 * Uses distance-based heuristics with element-specific covalent radii
 * Fallback when bond information is not available (e.g., from PDB without CONECT)
 *
 * @param {Array} atoms - Array of atom objects with {x, y, z, element}
 * @returns {Array} - Bond array in RDKit format: [{atoms: [i,j], bo: 1}, ...]
 */
function inferBondsFromCoordinates(atoms) {
  const bonds = [];
  const N = atoms.length;

  // Covalent radii in Angstroms (approximate)
  const covalentRadii = {
    'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57,
    'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Br': 1.20, 'I': 1.39
  };

  for (let i = 0; i < N; i++) {
    for (let j = i + 1; j < N; j++) {
      const dx = atoms[i].x - atoms[j].x;
      const dy = atoms[i].y - atoms[j].y;
      const dz = atoms[i].z - atoms[j].z;
      const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);

      // Get covalent radii for both atoms
      const r1 = covalentRadii[atoms[i].element] || 0.76; // Default to carbon
      const r2 = covalentRadii[atoms[j].element] || 0.76;

      // Bond if distance is within 1.3x sum of covalent radii
      const bondThreshold = (r1 + r2) * 1.3;

      if (dist < bondThreshold && dist > 0.4) { // Min distance to avoid noise
        bonds.push({ atoms: [i, j], bo: 1 }); // Assume single bond
      }
    }
  }

  console.log(`[inferBondsFromCoordinates] Inferred ${bonds.length} bonds from ${N} atoms`);
  return bonds;
}


/**
 * Encode target molecule using encoder network
 */
async function encodeTarget(atoms) {
  console.log('[encodeTarget] Starting encoding of', atoms.length, 'atoms');

  // Safety check: don't run if clearing
  if (isClearing) {
    throw new Error('Cannot encode during clear operation');
  }

  const N = atoms.length;

  // Extract positions and center at origin
  let positions = new Float32Array(N * 3);
  let sumX = 0, sumY = 0, sumZ = 0;

  for (let i = 0; i < N; i++) {
    sumX += atoms[i].x;
    sumY += atoms[i].y;
    sumZ += atoms[i].z;
  }

  const meanX = sumX / N;
  const meanY = sumY / N;
  const meanZ = sumZ / N;

  for (let i = 0; i < N; i++) {
    positions[i * 3 + 0] = atoms[i].x - meanX;
    positions[i * 3 + 1] = atoms[i].y - meanY;
    positions[i * 3 + 2] = atoms[i].z - meanZ;
  }

  // Get atomic numbers and features
  const atomicNums = atoms.map(a => getAtomicNumber(a.element));
  const features = atomicNumsToFeat(atomicNums);

  // Build edges using bond-based approach (matches training)
  // Infer bonds from atomic coordinates using covalent radii
  console.log('[encodeTarget] Inferring bonds from coordinates (covalent radii heuristic)...');
  const bonds = inferBondsFromCoordinates(atoms);
  const edgeIndex = buildBondEdgeIndex(bonds);

  // Validate edge indices
  const numEdges = edgeIndex.length / 2;
  console.log(`[encodeTarget] Built ${numEdges} bond-based edges for ${N} atoms (${bonds.length} bonds × 2 directions)`);

  // Check for out-of-bounds edge indices
  for (let i = 0; i < edgeIndex.length; i++) {
    const idx = Number(edgeIndex[i]);
    if (idx < 0 || idx >= N) {
      console.error(`[encodeTarget] OUT OF BOUNDS: Edge index ${i} has value ${idx}, but only ${N} atoms exist`);
      throw new Error(`Edge index out of bounds: ${idx} >= ${N}`);
    }
  }

  // Create batch tensor (all zeros for single molecule)
  const batch = new BigInt64Array(N).fill(0n);

  // Create ONNX tensors
  const xTensor = new ort.Tensor('float32', features, [N, DIFFMIP_CONFIG.nodeFeatDim]);
  const posTensor = new ort.Tensor('float32', positions, [N, 3]);
  const edgeTensor = new ort.Tensor('int64', edgeIndex, [2, edgeIndex.length / 2]);
  const batchTensor = new ort.Tensor('int64', batch, [N]);

  console.log(`[encodeTarget] Tensor shapes: x=${xTensor.dims}, pos=${posTensor.dims}, edges=${edgeTensor.dims}, batch=${batchTensor.dims}`);

  // Run encoder
  const feeds = {
    'x': xTensor,
    'pos': posTensor,
    'edge_index': edgeTensor,
    'batch': batchTensor
  };

  let results;
  try {
    console.log('[encodeTarget] Calling encoder.run()...');
    console.log('[encodeTarget] Model expects inputs:', diffmipModels.encoder.inputNames);
    console.log('[encodeTarget] We are providing:', Object.keys(feeds));
    results = await diffmipModels.encoder.run(feeds);
    console.log('[encodeTarget] Encoder inference completed successfully');
  } catch (error) {
    console.error('[encodeTarget] Encoder inference FAILED:', error);
    console.error('[encodeTarget] Error type:', error.constructor.name);
    console.error('[encodeTarget] Error message:', error.message);
    console.error('[encodeTarget] Error stack:', error.stack);
    console.error('[encodeTarget] Model expected inputs:', diffmipModels.encoder.inputNames);
    console.error('[encodeTarget] We provided inputs:', Object.keys(feeds));
    throw new Error(`Failed to encode target molecule: ${error.message}`);
  }

  // Convert positions back to 2D array format for compatibility
  const positionsArray = [];
  for (let i = 0; i < N; i++) {
    positionsArray.push([
      positions[i * 3 + 0],
      positions[i * 3 + 1],
      positions[i * 3 + 2]
    ]);
  }

  // Convert features to 2D array format
  const atomFeatures = [];
  for (let i = 0; i < N; i++) {
    const feat = [];
    for (let j = 0; j < DIFFMIP_CONFIG.nodeFeatDim; j++) {
      feat.push(features[i * DIFFMIP_CONFIG.nodeFeatDim + j]);
    }
    atomFeatures.push(feat);
  }

  return {
    nodeEmb: results.node_emb,
    globalCtx: results.global_ctx,
    batch: batchTensor,
    positions: positionsArray,  // Use 2D array
    atomFeatures: atomFeatures, // 2D array of features
    elements: atomicNums,       // Array of atomic numbers
    bonds: bonds                // Bond topology for consistency
  };
}


/**
 * Euler sampling for flow matching (continuous normalizing flow)
 */
async function eulerSampleCentroids(encodedTarget, numMonomers, numSteps) {
  console.log('[eulerSampleCentroids] Starting centroid sampling');

  // Safety check: don't run if clearing
  if (isClearing) {
    throw new Error('Cannot sample centroids during clear operation');
  }

  const B = 1; // Batch size
  const K = numMonomers;

  // Random monomer selection - limit to what model supports
  const maxTypes = Math.min(monomerLibrary.numTypes, MAX_MODEL_TYPES);

  const monoTypes = new BigInt64Array(K);
  for (let i = 0; i < K; i++) {
    monoTypes[i] = BigInt(Math.floor(Math.random() * maxTypes));
  }
  console.log(`[eulerSampleCentroids] Selected monomer types:`, Array.from(monoTypes).map(Number));
  console.log(`[eulerSampleCentroids] Monomer library has ${monomerLibrary.numTypes} types, model supports ${MAX_MODEL_TYPES} (using first ${maxTypes})`);

  // Validate monomer type indices
  for (let i = 0; i < K; i++) {
    const idx = Number(monoTypes[i]);
    if (idx < 0 || idx >= maxTypes) {
      console.error(`[eulerSampleCentroids] OUT OF BOUNDS: Monomer type ${i} has value ${idx}, but can only use ${maxTypes} types`);
      throw new Error(`Monomer type index out of bounds: ${idx} >= ${maxTypes}`);
    }
  }

  const monoTypesTensor = new ort.Tensor('int64', monoTypes, [B, K]);

  // Initialize random centroids (standard normal)
  let centroids = new Float32Array(B * K * 3);
  for (let i = 0; i < centroids.length; i++) {
    centroids[i] = randomNormal();
  }

  // Euler integration
  const dt = 1.0 / numSteps;

  for (let step = 0; step < numSteps; step++) {
    const t = step / numSteps;
    const tTensor = new ort.Tensor('float32', new Float32Array([t]), [B]);
    const centroidTensor = new ort.Tensor('float32', centroids, [B, K, 3]);

    // Predict velocity
    // Try to determine correct input names from model
    const modelInputs = diffmipModels.centroidNet.inputNames || [];
    const usesTargetPrefix = modelInputs.includes('target_node_emb');

    const feeds = usesTargetPrefix ? {
      't': tTensor,
      'centroid_t': centroidTensor,
      'mono_type_idx': monoTypesTensor,
      'target_node_emb': encodedTarget.nodeEmb,
      'target_global_ctx': encodedTarget.globalCtx,
      'target_batch': encodedTarget.batch
    } : {
      't': tTensor,
      'centroid_t': centroidTensor,
      'mono_type_idx': monoTypesTensor,
      'node_emb': encodedTarget.nodeEmb,
      'global_ctx': encodedTarget.globalCtx,
      'batch': encodedTarget.batch
    };

    // Log tensor shapes at step 0 for debugging
    if (step === 0) {
      console.log(`[eulerSampleCentroids] Input tensor shapes at step 0:`);
      console.log(`  t: ${tTensor.dims} (value: ${t})`);
      console.log(`  centroid_t: ${centroidTensor.dims}`);
      console.log(`  mono_type_idx: ${monoTypesTensor.dims}`);
      console.log(`  node_emb: ${encodedTarget.nodeEmb.dims}`);
      console.log(`  global_ctx: ${encodedTarget.globalCtx.dims}`);
      console.log(`  batch: ${encodedTarget.batch.dims}`);
      console.log(`[eulerSampleCentroids] Model: ${diffmipModels.centroidNet ? 'loaded' : 'NOT LOADED'}`);

      // Log expected input names from model
      if (diffmipModels.centroidNet && diffmipModels.centroidNet.inputNames) {
        console.log(`[eulerSampleCentroids] Model expects inputs:`, diffmipModels.centroidNet.inputNames);
      }

      // Log naming convention detected
      console.log(`[eulerSampleCentroids] Using ${usesTargetPrefix ? 'NEW' : 'OLD'} naming convention (${usesTargetPrefix ? 'target_*' : 'no prefix'})`);

      // Log what we're providing
      console.log(`[eulerSampleCentroids] We are providing:`, Object.keys(feeds));
    }

    let results;
    try {
      console.log(`[eulerSampleCentroids] Step ${step}/${numSteps}: calling centroidNet.run()...`);
      results = await diffmipModels.centroidNet.run(feeds);
      console.log(`[eulerSampleCentroids] Step ${step}/${numSteps}: completed successfully`);
    } catch (error) {
      console.error(`[eulerSampleCentroids] Centroid network inference FAILED at step ${step}:`, error);
      console.error(`[eulerSampleCentroids] Error type:`, error.constructor.name);
      console.error(`[eulerSampleCentroids] Error message:`, error.message);
      console.error(`[eulerSampleCentroids] Error stack:`, error.stack);
      console.error(`[eulerSampleCentroids] Input shapes were:`);
      console.error(`  t: ${tTensor.dims}`);
      console.error(`  centroid_t: ${centroidTensor.dims}`);
      console.error(`  mono_type_idx: ${monoTypesTensor.dims}`);
      console.error(`  target_node_emb: ${encodedTarget.nodeEmb.dims}`);
      console.error(`  target_global_ctx: ${encodedTarget.globalCtx.dims}`);
      console.error(`  target_batch: ${encodedTarget.batch.dims}`);
      throw new Error(`Centroid sampling failed at step ${step}: ${error.message}`);
    }

    const velocity = results.velocity.data;

    // Update centroids: x_{t+dt} = x_t + v_t * dt
    for (let i = 0; i < centroids.length; i++) {
      centroids[i] += velocity[i] * dt;
    }
  }

  return { centroids, monoTypes };
}


/**
 * Euler sampling for torsions
 */
async function eulerSampleTorsions(encodedTarget, centroids, monoTypes, numMonomers, numSteps) {
  console.log('[eulerSampleTorsions] Starting torsion sampling');

  // Safety check: don't run if clearing
  if (isClearing) {
    throw new Error('Cannot sample torsions during clear operation');
  }

  const B = 1;
  const K = numMonomers;
  const D = DIFFMIP_CONFIG.maxDof;

  // For simplicity, use uniform DOF mask (all enabled)
  // In practice, this should be computed per monomer based on rotatable bonds
  const dofMask = new Float32Array(B * K * D).fill(1.0);
  const dofMaskTensor = new ort.Tensor('float32', dofMask, [B, K, D]);

  // Initialize random DOFs
  let dofs = new Float32Array(B * K * D);
  for (let i = 0; i < dofs.length; i++) {
    dofs[i] = randomNormal();
  }

  // Convert types to tensor if not already
  const monoTypesTensor = new ort.Tensor('int64', monoTypes, [B, K]);
  const centroidsTensor = new ort.Tensor('float32', centroids, [B, K, 3]);

  // Euler integration
  const dt = 1.0 / numSteps;

  for (let step = 0; step < numSteps; step++) {
    const t = step / numSteps;
    const sigmaTensor = new ort.Tensor('float32', new Float32Array([t]), [B]);
    const dofTensor = new ort.Tensor('float32', dofs, [B, K, D]);

    // Predict velocity
    // Try to determine correct input names from model
    const modelInputs = diffmipModels.torsionNet.inputNames || [];
    const usesTargetPrefix = modelInputs.includes('target_node_emb');
    const usesSigma = modelInputs.includes('sigma');

    const feeds = usesTargetPrefix ? {
      [usesSigma ? 'sigma' : 't']: sigmaTensor,
      'dof_t': dofTensor,
      'dof_mask': dofMaskTensor,
      'centroid_0': centroidsTensor,
      'mono_type_idx': monoTypesTensor,
      'target_node_emb': encodedTarget.nodeEmb,
      'target_global_ctx': encodedTarget.globalCtx,
      'target_batch': encodedTarget.batch
    } : {
      [usesSigma ? 'sigma' : 't']: sigmaTensor,
      'dof_t': dofTensor,
      'dof_mask': dofMaskTensor,
      'centroid_0': centroidsTensor,
      'mono_type_idx': monoTypesTensor,
      'node_emb': encodedTarget.nodeEmb,
      'global_ctx': encodedTarget.globalCtx,
      'batch': encodedTarget.batch
    };

    // Log at step 0 for debugging
    if (step === 0) {
      console.log(`[eulerSampleTorsions] Input tensor shapes at step 0:`);
      if (diffmipModels.torsionNet && diffmipModels.torsionNet.inputNames) {
        console.log(`[eulerSampleTorsions] Model expects inputs:`, diffmipModels.torsionNet.inputNames);
      }
      console.log(`[eulerSampleTorsions] Using ${usesTargetPrefix ? 'NEW' : 'OLD'} naming convention (${usesTargetPrefix ? 'target_*' : 'no prefix'})`);
      console.log(`[eulerSampleTorsions] Using ${usesSigma ? 'sigma' : 't'} for timestep`);
      console.log(`[eulerSampleTorsions] We are providing:`, Object.keys(feeds));
    }

    let results;
    try {
      console.log(`[eulerSampleTorsions] Step ${step}/${numSteps}: calling torsionNet.run()...`);
      results = await diffmipModels.torsionNet.run(feeds);
      console.log(`[eulerSampleTorsions] Step ${step}/${numSteps}: completed successfully`);
    } catch (error) {
      console.error(`[eulerSampleTorsions] Torsion network inference FAILED at step ${step}:`, error);
      console.error(`[eulerSampleTorsions] Error type:`, error.constructor.name);
      console.error(`[eulerSampleTorsions] Error message:`, error.message);
      console.error(`[eulerSampleTorsions] Error stack:`, error.stack);
      console.error(`[eulerSampleTorsions] Model expected inputs:`, diffmipModels.torsionNet.inputNames);
      console.error(`[eulerSampleTorsions] We provided inputs:`, Object.keys(feeds));
      throw new Error(`Torsion sampling failed at step ${step}: ${error.message}`);
    }

    const velocity = results.velocity.data;

    // Update DOFs
    for (let i = 0; i < dofs.length; i++) {
      dofs[i] += velocity[i] * dt;
    }
  }

  return dofs;
}


/**
 * Generate random number from standard normal distribution (Box-Muller transform)
 */
function randomNormal() {
  const u1 = Math.random();
  const u2 = Math.random();
  return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
}


// =============================================================================
// RECIPE MODE: Iterative monomer placement
// =============================================================================

/**
 * Class to track placed monomers during iterative building
 */
class PlacedMonomer {
  constructor(code, monomerIdx, smiles, positions, elements, centroid, euler, torsions, targetScore, monomerScores, structureText) {
    this.code = code;
    this.monomerIdx = monomerIdx;
    this.smiles = smiles;
    this.positions = positions;  // Heavy atom 3D coordinates
    this.elements = elements;
    this.centroid = centroid;
    this.euler = euler;
    this.torsions = torsions;
    this.targetScore = targetScore;  // Energy vs target
    this.monomerScores = monomerScores;  // Energies vs other placed monomers
    this.structureText = structureText;  // PDB text for visualization
  }
}

/**
 * Create atom features from JSON data (RDKit MinimalLib doesn't expose get_atom)
 */
function getAtomFeaturesFromJSON(atomData, atomicNum, bonds, atomIdx) {
  const ATOM_TYPES = [1, 6, 7, 8, 9, 15, 16, 17, 35];
  const NUM_ATOM_TYPES = 10;

  // Atom type one-hot
  const typeIdx = ATOM_TYPES.indexOf(atomicNum);
  const typeVec = new Array(NUM_ATOM_TYPES).fill(0);
  typeVec[typeIdx >= 0 ? typeIdx : NUM_ATOM_TYPES - 1] = 1;

  // Formal charge from JSON
  const charge = atomData.chg || 0;

  // Compute degree from bonds
  let degree = 0;
  for (const bond of bonds) {
    if (bond.atoms && (bond.atoms[0] === atomIdx || bond.atoms[1] === atomIdx)) {
      degree++;
    }
  }
  degree = Math.min(5, degree);
  const degreeVec = new Array(6).fill(0);
  degreeVec[degree] = 1;

  // Estimate implicit H count based on valence rules
  const valenceMap = { 1: 1, 6: 4, 7: 3, 8: 2, 9: 1, 15: 5, 16: 2, 17: 1, 35: 1 };
  const maxValence = valenceMap[atomicNum] || 4;
  let hCount = Math.max(0, maxValence - degree);
  hCount = Math.min(4, hCount);
  const hVec = new Array(5).fill(0);
  hVec[hCount] = 1;

  // Aromatic and ring info not available from JSON in MinimalLib
  const inRing = 0;
  const aromatic = 0;

  return [
    ...typeVec,       // 10 dims
    charge,           // 1 dim
    inRing,           // 1 dim
    aromatic,         // 1 dim
    ...hVec,          // 5 dims
    ...degreeVec,     // 6 dims
  ];
}

/**
 * Extract 24-dimensional atom features from RDKit.js molecule
 *
 * NOTE: This function is NOT USED because RDKit MinimalLib doesn't expose get_atom().
 * We use getAtomFeaturesFromJSON() instead which extracts from JSON data.
 * Keeping this for reference in case a future RDKit version adds get_atom().
 *
 * Features:
 * - 10 dims: one-hot atom type (H, C, N, O, F, P, S, Cl, Br, other)
 * - 1 dim: formal charge (clipped -2 to 2)
 * - 1 dim: is in ring
 * - 1 dim: is aromatic
 * - 5 dims: number of implicit H (0-4) one-hot
 * - 6 dims: degree (0-5) one-hot
 * Total: 24 dims
 */
function getAtomFeatures(mol, atomIdx) {
  const ATOM_TYPES = [1, 6, 7, 8, 9, 15, 16, 17, 35]; // H C N O F P S Cl Br
  const NUM_ATOM_TYPES = 10; // 9 + 1 for "other"

  // Get atom (not available in MinimalLib)
  const atom = mol.get_atom(atomIdx);
  const atomicNum = atom.get_atomic_num();

  // Atom type one-hot
  const typeIdx = ATOM_TYPES.indexOf(atomicNum);
  const typeVec = new Array(NUM_ATOM_TYPES).fill(0);
  typeVec[typeIdx >= 0 ? typeIdx : NUM_ATOM_TYPES - 1] = 1;

  // Formal charge (clipped to [-2, 2])
  let charge = atom.get_formal_charge();
  charge = Math.max(-2, Math.min(2, charge));

  // Is in ring?
  const inRing = atom.is_in_ring() ? 1 : 0;

  // Is aromatic?
  const aromatic = atom.get_is_aromatic() ? 1 : 0;

  // Implicit H count one-hot (0-4)
  let hCount = atom.get_total_num_hs();
  hCount = Math.max(0, Math.min(4, hCount));
  const hVec = new Array(5).fill(0);
  hVec[hCount] = 1;

  // Degree one-hot (0-5)
  let degree = atom.get_degree();
  degree = Math.max(0, Math.min(5, degree));
  const degreeVec = new Array(6).fill(0);
  degreeVec[degree] = 1;

  // Concatenate all features
  return [
    ...typeVec,       // 10 dims
    charge,           // 1 dim
    inRing,           // 1 dim
    aromatic,         // 1 dim
    ...hVec,          // 5 dims
    ...degreeVec      // 6 dims
  ]; // Total: 24 dims
}

/**
 * Build radius-based edge index for a molecular graph
 * @param {Array<Array<number>>} positions - Nx3 array of atom positions
 * @param {number} maxRadius - Maximum distance for edge connection (default: 5.0 Angstroms)
 * @returns {Int32Array} - Edge index as [2, E] flattened COO format
 */
function buildEdgeIndex(positions, maxRadius = 5.0) {
  const N = positions.length;
  const edges = [];

  for (let i = 0; i < N; i++) {
    for (let j = i + 1; j < N; j++) {
      const dx = positions[i][0] - positions[j][0];
      const dy = positions[i][1] - positions[j][1];
      const dz = positions[i][2] - positions[j][2];
      const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);

      if (dist < maxRadius && dist > 0.01) {
        // Add bidirectional edges (i->j and j->i)
        edges.push([i, j]);
        edges.push([j, i]);
      }
    }
  }

  // Convert to [2, E] format
  const edgeIndex = new Int32Array(edges.length * 2);
  for (let e = 0; e < edges.length; e++) {
    edgeIndex[e] = edges[e][0];                // Source indices
    edgeIndex[edges.length + e] = edges[e][1]; // Target indices
  }

  return edgeIndex;
}

/**
 * Physics-based screening: score all monomers against target
 * Matches diffmip-lite's screen_monomers_physics function
 *
 * For each monomer, samples multiple random orientations around the target,
 * scores each with physics-based energy, and takes the best score.
 *
 * @param {Object} encodedTarget - Encoded target with positions and elements
 * @param {number} topK - Number of top candidates to return
 * @param {number} numOrientations - Number of random orientations to try per monomer
 * @returns {Array} - Top-k candidates with idx, code, and score
 */
async function screenMonomersPhysics(encodedTarget, topK = 5, numOrientations = 5) {
  // IMPORTANT: Only screen monomers the model can handle
  const maxMonomerIdx = Math.min(monomerLibrary.monomers.length, MAX_MODEL_TYPES);
  console.log(`[screenMonomersPhysics] Screening first ${maxMonomerIdx} of ${monomerLibrary.monomers.length} monomers (model limit: ${MAX_MODEL_TYPES}) with ${numOrientations} orientations each...`);

  const results = [];
  const targetPos = encodedTarget.positions;
  const targetElements = encodedTarget.elements;

  for (let idx = 0; idx < maxMonomerIdx; idx++) {
    const monomer = monomerLibrary.monomers[idx];
    if (!monomer.smiles) continue;

    try {
      // Generate structure for monomer
      const result = generateStructureFromSMILES(monomer.smiles);
      if (!result) continue;

      // Center monomer at origin
      const centroid = result.coords.reduce((acc, [x, y, z]) => [
        acc[0] + x / result.coords.length,
        acc[1] + y / result.coords.length,
        acc[2] + z / result.coords.length
      ], [0, 0, 0]);

      const monoCentered = result.coords.map(([x, y, z]) => [
        x - centroid[0],
        y - centroid[1],
        z - centroid[2]
      ]);

      // Extract elements from structure
      const monoElements = result.elements || result.coords.map(() => 6);

      let bestScore = Infinity;

      // Try multiple random orientations
      for (let orient = 0; orient < numOrientations; orient++) {
        // Random Euler angles
        const alpha = (Math.random() * 2 - 1) * Math.PI;
        const beta = (Math.random() * 2 - 1) * Math.PI;
        const gamma = (Math.random() * 2 - 1) * Math.PI;

        // Rotation matrices
        const cosA = Math.cos(alpha), sinA = Math.sin(alpha);
        const cosB = Math.cos(beta), sinB = Math.sin(beta);
        const cosG = Math.cos(gamma), sinG = Math.sin(gamma);

        const Rx = [
          [1, 0, 0],
          [0, cosA, -sinA],
          [0, sinA, cosA]
        ];

        const Ry = [
          [cosB, 0, sinB],
          [0, 1, 0],
          [-sinB, 0, cosB]
        ];

        const Rz = [
          [cosG, -sinG, 0],
          [sinG, cosG, 0],
          [0, 0, 1]
        ];

        // Combined rotation: R = Rz @ Ry @ Rx
        const matMul = (A, B) => {
          return [
            [A[0][0]*B[0][0] + A[0][1]*B[1][0] + A[0][2]*B[2][0],
             A[0][0]*B[0][1] + A[0][1]*B[1][1] + A[0][2]*B[2][1],
             A[0][0]*B[0][2] + A[0][1]*B[1][2] + A[0][2]*B[2][2]],
            [A[1][0]*B[0][0] + A[1][1]*B[1][0] + A[1][2]*B[2][0],
             A[1][0]*B[0][1] + A[1][1]*B[1][1] + A[1][2]*B[2][1],
             A[1][0]*B[0][2] + A[1][1]*B[1][2] + A[1][2]*B[2][2]],
            [A[2][0]*B[0][0] + A[2][1]*B[1][0] + A[2][2]*B[2][0],
             A[2][0]*B[0][1] + A[2][1]*B[1][1] + A[2][2]*B[2][1],
             A[2][0]*B[0][2] + A[2][1]*B[1][2] + A[2][2]*B[2][2]]
          ];
        };

        const R = matMul(matMul(Rz, Ry), Rx);

        // Random position around target (6-10 Angstrom radius)
        const direction = [
          Math.random() * 2 - 1,
          Math.random() * 2 - 1,
          Math.random() * 2 - 1
        ];
        const dirLen = Math.sqrt(direction[0]**2 + direction[1]**2 + direction[2]**2) || 1e-8;
        const dirNorm = direction.map(d => d / dirLen);
        const radius = 6.0 + Math.random() * 4.0; // 6-10 Å
        const offset = dirNorm.map(d => d * radius);

        // Apply transformation: mono_transformed = (mono_centered @ R.T) + offset
        const monoTransformed = monoCentered.map(([x, y, z]) => {
          const rotated = [
            R[0][0]*x + R[1][0]*y + R[2][0]*z,
            R[0][1]*x + R[1][1]*y + R[2][1]*z,
            R[0][2]*x + R[1][2]*y + R[2][2]*z
          ];
          return [
            rotated[0] + offset[0],
            rotated[1] + offset[1],
            rotated[2] + offset[2]
          ];
        });

        // Score with simplified distance-based energy
        // This is a fast approximation for screening
        let interactionScore = 0;
        let numInteractions = 0;

        for (let i = 0; i < targetPos.length; i++) {
          for (let j = 0; j < monoTransformed.length; j++) {
            const dx = targetPos[i][0] - monoTransformed[j][0];
            const dy = targetPos[i][1] - monoTransformed[j][1];
            const dz = targetPos[i][2] - monoTransformed[j][2];
            const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);

            // Only consider close contacts (< 5 Å)
            if (dist < 5.0 && dist > 0.5) {
              // Simple LJ-like potential: favor contacts around 3-4 Å
              const optimalDist = 3.5;
              const score_contrib = -1.0 / (1.0 + Math.abs(dist - optimalDist));
              interactionScore += score_contrib;
              numInteractions++;
            }
          }
        }

        // Size-corrected energy (normalize by number of atoms)
        const combinedSize = targetPos.length + monoTransformed.length;
        const size_corrected_energy = numInteractions > 0
          ? interactionScore / combinedSize
          : 0.0;

        if (size_corrected_energy < bestScore) {
          bestScore = size_corrected_energy;
        }
      }

      results.push({
        idx: idx,
        code: monomer.code,
        score: bestScore
      });

    } catch (error) {
      console.warn(`[screenMonomersPhysics] Failed to screen ${monomer.code}:`, error.message);
    }
  }

  // Sort by score (most negative = best)
  results.sort((a, b) => a.score - b.score);

  const topCandidates = results.slice(0, topK);
  console.log(`[screenMonomersPhysics] Top ${topK} candidates:`,
    topCandidates.map(c => `${c.code} (${c.score.toFixed(2)})`));

  return topCandidates;
}

/**
 * Screen monomers against target using ScreeningNet (neural network)
 * Returns top-k candidates sorted by predicted binding affinity
 * @param {Object} encodedTarget - Encoded target molecule with positions
 * @param {number} topK - Number of top candidates to return
 * @returns {Array} - Top-k candidates with idx, code, and score
 */
async function screenMonomersNeural(encodedTarget, topK = 5) {
  console.log(`[screenMonomersNeural] Screening ${monomerLibrary.monomers.length} monomers with neural network...`);

  const candidates = [];
  const maxRadius = DIFFMIP_CONFIG.maxRadius || 5.0;

  // Get target data
  const targetPos = encodedTarget.positions;
  const targetX = encodedTarget.atomFeatures; // Use proper atom features
  const targetBatch = new BigInt64Array(targetPos.length).fill(0n);

  // Build target edge index from bonds (bond-based graph, not spatial radius)
  if (!encodedTarget.bonds) {
    throw new Error('[screenMonomersNeural] encodedTarget missing bonds - ensure encodeTarget() returns bond information');
  }
  const targetEdgeIndex = buildBondEdgeIndex(encodedTarget.bonds);
  console.log(`[screenMonomersNeural] Target: ${targetPos.length} atoms, ${encodedTarget.bonds.length} bonds, ${targetEdgeIndex.length / 2} edges`);

  // Get loading message and status elements for progress updates
  const loadingMsg = document.querySelector('#loading p');
  const statusEl = $('#diffmip-status');

  // Screen each monomer
  for (let idx = 0; idx < monomerLibrary.monomers.length; idx++) {
    const monomer = monomerLibrary.monomers[idx];

    // Update progress every 5 monomers
    if (idx % 5 === 0) {
      const progressPercent = (idx / monomerLibrary.monomers.length) * 100;
      const progress = `Screening monomer ${idx + 1} of ${monomerLibrary.monomers.length}`;
      updateProgress('Screening Monomers', progress, progressPercent);
      if (statusEl) statusEl.text(progress).css('color', '#fbbf24');
    }

    if (!monomer.smiles) continue;

    try {
      // Generate structure for monomer
      const result = generateStructureFromSMILES(monomer.smiles);
      if (!result) continue;

      // Center monomer at origin
      const centroid = result.coords.reduce((acc, [x, y, z]) => [
        acc[0] + x / result.coords.length,
        acc[1] + y / result.coords.length,
        acc[2] + z / result.coords.length
      ], [0, 0, 0]);

      const monoPos = result.coords.map(([x, y, z]) => [
        x - centroid[0],
        y - centroid[1],
        z - centroid[2]
      ]);

      const monoX = result.atomFeatures; // Use proper atom features from structure
      const monoBatch = new BigInt64Array(monoPos.length).fill(0n);

      // Build monomer edge index from bonds (bond-based graph, not spatial radius)
      if (!result.bonds) {
        console.warn(`[screenMonomersNeural] Monomer ${monomer.code} missing bonds - skipping`);
        continue;
      }
      const monoEdgeIndex = buildBondEdgeIndex(result.bonds);

      // Flatten positions and features for ONNX
      const targetPosFlat = new Float32Array(targetPos.flat());
      const targetXFlat = new Float32Array(targetX.flat());
      const monoPosFlat = new Float32Array(monoPos.flat());
      const monoXFlat = new Float32Array(monoX.flat());

      // Run screening model with edge indices (all int64)
      const feeds = {
        'target_x': new ort.Tensor('float32', targetXFlat, [targetPos.length, DIFFMIP_CONFIG.nodeFeatDim]),
        'target_pos': new ort.Tensor('float32', targetPosFlat, [targetPos.length, 3]),
        'target_edge_index': new ort.Tensor('int64', targetEdgeIndex, [2, targetEdgeIndex.length / 2]),
        'target_batch': new ort.Tensor('int64', targetBatch, [targetPos.length]),
        'mono_x': new ort.Tensor('float32', monoXFlat, [monoPos.length, DIFFMIP_CONFIG.nodeFeatDim]),
        'mono_pos': new ort.Tensor('float32', monoPosFlat, [monoPos.length, 3]),
        'mono_edge_index': new ort.Tensor('int64', monoEdgeIndex, [2, monoEdgeIndex.length / 2]),
        'mono_batch': new ort.Tensor('int64', monoBatch, [monoPos.length])
      };

      const results = await diffmipModels.screeningNet.run(feeds);
      const energyPred = results.energy.data[0];  // Standardized energy prediction

      if (idx < 5) {
        console.log(`[screenMonomersNeural] ${monomer.code}: energy=${energyPred.toFixed(2)}, edges=${monoEdgeIndex.length/2}`);
      }

      candidates.push({
        idx: idx,
        code: monomer.code,
        score: energyPred
      });

    } catch (error) {
      console.warn(`[screenMonomers] Failed to screen ${monomer.code}:`, error.message);
    }
  }

  console.log(`[screenMonomersNeural] Screened ${candidates.length} monomers successfully`);

  const screeningComplete = `Screening complete: ${candidates.length} monomers scored`;
  updateProgress('Screening Complete', screeningComplete, 100);
  if (statusEl) statusEl.text(screeningComplete).css('color', '#22c55e');

  if (candidates.length === 0) {
    console.warn('[screenMonomersNeural] No candidates found! All monomers may be missing bonds.');
    const failMsg = 'Screening failed: no valid monomers found';
    updateProgress('Screening Failed', failMsg, 100);
    if (statusEl) statusEl.text(failMsg).css('color', '#f87171');
    return [];
  }

  // Sort by score (lower = better binding)
  candidates.sort((a, b) => a.score - b.score);

  // Show score distribution
  const scores = candidates.map(c => c.score);
  const minScore = Math.min(...scores);
  const maxScore = Math.max(...scores);
  const meanScore = scores.reduce((a, b) => a + b, 0) / scores.length;
  console.log(`[screenMonomersNeural] Score range: ${minScore.toFixed(2)} to ${maxScore.toFixed(2)}, mean: ${meanScore.toFixed(2)}`);

  // Return top-k
  const topCandidates = candidates.slice(0, topK);
  console.log(`[screenMonomersNeural] Top ${topK} candidates:`, topCandidates.map(c => `${c.code} (${c.score.toFixed(2)})`));

  if (loadingMsg) {
    loadingMsg.textContent = `Top ${topK} candidates selected`;
  }

  return topCandidates;
}

/**
 * Screen monomers - wrapper function
 *
 * Uses neural screening (SimpleScreeningNet) if loaded, otherwise falls back
 * to physics-based screening with random orientations.
 *
 * Neural screening is much faster and more faithful to the original diffMIP approach:
 * - Single forward pass per monomer (~1-2ms each)
 * - Predicts energy + H-bond count from graph structure
 * - No random orientation sampling
 *
 * @param {Object} encodedTarget - Encoded target with positions, elements, atomFeatures
 * @param {number} topK - Number of top candidates to return
 * @param {number} numOrientations - Number of random orientations (for physics fallback)
 * @returns {Array} - Top-k candidates
 */
async function screenMonomers(encodedTarget, topK = 5, numOrientations = 5) {
  // Use neural screening if model is loaded
  if (diffmipModels.screeningNet) {
    console.log('[screenMonomers] Using neural screening (SimpleScreeningNet)');
    return await screenMonomersNeural(encodedTarget, topK);
  } else {
    console.log('[screenMonomers] Screening model not loaded, using physics-based fallback');
    return await screenMonomersPhysics(encodedTarget, topK, numOrientations);
  }
}

/**
 * Iteratively build receptor around target molecule (recipe.py approach)
 *
 * Algorithm:
 * 1. For each iteration:
 *    a. SCREEN all monomers → get top-k candidates
 *    b. For each candidate, generate N placements
 *    c. Score each placement against target AND placed monomers
 *    d. Select single best placement
 *    e. Add to placed list
 * 2. Stop when:
 *    - Max monomers reached
 *    - No good placements found
 *    - New monomer interacts more with placed than target
 */
async function buildReceptorIteratively(
  maxMonomers = 5,
  samplesPerMonomer = 10,
  topKCandidates = 5,
  energyThreshold = -3.0
) {
  console.log(`[buildReceptorIteratively] Starting iterative building: max=${maxMonomers}, samples=${samplesPerMonomer}, topK=${topKCandidates}`);

  const placedMonomers = [];
  const loadingMsg = document.querySelector('#loading p');
  const statusEl = $('#diffmip-status');

  for (let iteration = 0; iteration < maxMonomers; iteration++) {
    console.log(`\n${'='.repeat(70)}`);
    console.log(`[Recipe] Iteration ${iteration + 1}/${maxMonomers}`);
    console.log(`${'='.repeat(70)}`);

    const iterationMsg = `Building receptor (${iteration + 1}/${maxMonomers})...`;
    const iterationProgress = (iteration / maxMonomers) * 100;
    updateProgress('Recipe Building', iterationMsg, iterationProgress);
    if (loadingMsg) loadingMsg.textContent = iterationMsg;
    if (statusEl) statusEl.text(iterationMsg).css('color', '#fbbf24');

    // STEP 1: Screen monomers to get top-k candidates
    console.log(`[Recipe] Step 1: Screening monomers...`);
    const encodedTarget = await encodeTargetMolecule();
    const candidates = await screenMonomers(encodedTarget, topKCandidates);

    if (candidates.length === 0) {
      console.warn(`[Recipe] No candidates from screening`);
      break;
    }

    let bestPlacement = null;
    let bestScore = Infinity;
    let bestCandidateIdx = -1;

    // STEP 2: Try each candidate monomer type
    console.log(`[Recipe] Step 2: Generating placements for ${candidates.length} candidates...`);
    for (let candIdx = 0; candIdx < candidates.length; candIdx++) {
      const candidate = candidates[candIdx];
      const monomer = monomerLibrary.monomers[candidate.idx];
      console.log(`[Recipe] Trying ${monomer.code} (${candIdx + 1}/${candidates.length}, screening score: ${candidate.score.toFixed(2)})...`);

      const candidateMsg = `Iteration ${iteration + 1}: trying ${monomer.code} (${candIdx + 1}/${candidates.length})`;
      updateProgress('Generating Placements', candidateMsg, (candIdx / candidates.length) * 100);
      if (loadingMsg) {
        loadingMsg.textContent = `${candidateMsg} (${samplesPerMonomer} samples)...`;
      }

      // Generate placements for this monomer
      try {
        const placements = await generatePlacementsForMonomer(
          candidate.idx,  // Use actual monomer index from screening
          samplesPerMonomer
        );

        console.log(`[Recipe] Generated ${placements.length} placements for ${monomer.code}`);

        if (placements.length === 0) {
          console.warn(`[Recipe] No placements generated for ${monomer.code} - skipping`);
          continue;
        }

        // Score each placement
        let validPlacementsCount = 0;
        let scoredCount = 0;
        const allScores = [];
        for (let pIdx = 0; pIdx < placements.length; pIdx++) {
          const placement = placements[pIdx];

          // Update progress
          if (pIdx % 3 === 0) {
            const scoreMsg = `Scoring ${monomer.code} placements: ${pIdx + 1}/${placements.length}...`;
            const scoreProgress = (pIdx / placements.length) * 100;
            updateProgress('Scoring Placements', scoreMsg, scoreProgress);
            if (loadingMsg) loadingMsg.textContent = scoreMsg;
            if (statusEl) statusEl.text(scoreMsg).css('color', '#fbbf24');
          }

          // Compute structure and score against target
          const targetScore = await scorePlacement(placement);

          if (!targetScore) {
            continue;  // Scoring failed
          }

          scoredCount++;
          allScores.push(targetScore.energy_total);

          if (targetScore.energy_total >= energyThreshold) {
            continue;  // Skip placements above threshold (using energy_total)
          }

          validPlacementsCount++;

          // Score against all placed monomers (PARALLELIZED)
          const monomerScorePromises = placedMonomers.map(placed =>
            scorePlacementAgainstPlaced(placement, placed)
          );
          const monomerScoreResults = await Promise.all(monomerScorePromises);
          const monomerScores = monomerScoreResults
            .filter(score => score !== null)
            .map(score => score.energy_total);  // Use energy_total

          // Combined score: target interaction + penalty for monomer overlaps
          const totalMonomerScore = monomerScores.reduce((sum, s) => sum + s, 0);
          const combinedScore = targetScore.energy_total + 0.5 * totalMonomerScore;  // Use energy_total

          if (combinedScore < bestScore) {
            bestScore = combinedScore;
            bestPlacement = {
              ...placement,
              targetScore: targetScore,
              monomerScores: monomerScores,
              totalMonomerScore: totalMonomerScore
            };
            bestCandidateIdx = candidate.idx;  // Use actual monomer index
          }
        }

        // Log statistics for this candidate
        if (allScores.length > 0) {
          const minE = Math.min(...allScores);
          const maxE = Math.max(...allScores);
          const meanE = allScores.reduce((a, b) => a + b, 0) / allScores.length;
          console.log(`[Recipe] ${monomer.code}: ${scoredCount}/${placements.length} scored, energies: ${minE.toFixed(1)} to ${maxE.toFixed(1)} (mean ${meanE.toFixed(1)}), ${validPlacementsCount} passed threshold ${energyThreshold}`);
        } else {
          console.warn(`[Recipe] ${monomer.code}: No placements could be scored`);
        }

      } catch (error) {
        console.error(`[Recipe] Error processing ${monomer.code}:`, error);
      }
    }

    // Check if we found any valid placement
    if (!bestPlacement) {
      console.log(`[Recipe] ⚠️  No valid placements found (energy < ${energyThreshold} kcal/mol)`);
      console.log(`[Recipe] Terminating at ${placedMonomers.length} monomers`);
      const terminateMsg = `Complete: placed ${placedMonomers.length} monomers (no more valid placements)`;
      if (loadingMsg) loadingMsg.textContent = terminateMsg;
      if (statusEl) statusEl.text(terminateMsg).css('color', '#fbbf24');
      break;
    }

    const bestMonomer = monomerLibrary.monomers[bestCandidateIdx];
    console.log(`[Recipe] ✓ Best placement: ${bestMonomer.code}`);
    console.log(`[Recipe]   Target score: ${bestPlacement.targetScore.energy_total.toFixed(2)} kcal/mol`);
    console.log(`[Recipe]   Monomer score: ${bestPlacement.totalMonomerScore.toFixed(2)} kcal/mol`);

    const placedMsg = `Iteration ${iteration + 1}: placed ${bestMonomer.code} (E=${bestPlacement.targetScore.energy_total.toFixed(1)} kcal/mol)`;
    const placedProgress = ((iteration + 1) / maxMonomers) * 100;
    updateProgress('Monomer Placed', placedMsg, placedProgress);
    if (loadingMsg) loadingMsg.textContent = placedMsg;
    if (statusEl) statusEl.text(placedMsg).css('color', '#22c55e');
    console.log(`[Recipe]   Combined: ${bestScore.toFixed(2)} kcal/mol`);

    // Check termination criterion: stop if new monomer interacts more with placed than target
    if (placedMonomers.length > 0 &&
        Math.abs(bestPlacement.totalMonomerScore) > Math.abs(bestPlacement.targetScore.energy_total)) {
      console.log(`[Recipe] Terminating: monomer interactions (${Math.abs(bestPlacement.totalMonomerScore).toFixed(2)}) > target (${Math.abs(bestPlacement.targetScore.energy_total).toFixed(2)})`);
      break;
    }

    // Add to placed list
    const placed = new PlacedMonomer(
      bestMonomer.code,
      bestCandidateIdx,
      bestMonomer.smiles,
      bestPlacement.positions,
      bestPlacement.elements,
      bestPlacement.centroid,
      bestPlacement.euler,
      bestPlacement.torsions,
      bestPlacement.targetScore,
      bestPlacement.monomerScores,
      bestPlacement.structureText
    );
    placedMonomers.push(placed);

    console.log(`[Recipe] ✓ Placed ${bestMonomer.code} (total: ${placedMonomers.length})`);
  }

  console.log(`\n[Recipe] Building complete: ${placedMonomers.length} monomers placed`);

  const completeMsg = `Recipe complete: ${placedMonomers.length} monomers placed successfully`;
  updateProgress('Recipe Complete', completeMsg, 100);
  if (loadingMsg) loadingMsg.textContent = completeMsg;
  if (statusEl) statusEl.text(completeMsg).css('color', '#22c55e');

  return placedMonomers;
}

/**
 * Helper: Encode the target molecule (cached)
 */
let cachedEncodedTarget = null;
async function encodeTargetMolecule() {
  if (cachedEncodedTarget) {
    return cachedEncodedTarget;
  }

  // Parse and encode target
  const atoms = parsePDB(rawPdbText);
  cachedEncodedTarget = await encodeTarget(atoms);
  return cachedEncodedTarget;
}

/**
 * Generate multiple placements for a single monomer type
 */
async function generatePlacementsForMonomer(monomerIdx, numSamples) {
  // Validate monomer index is within model's supported range
  if (monomerIdx >= MAX_MODEL_TYPES) {
    console.error(`[generatePlacementsForMonomer] Monomer index ${monomerIdx} exceeds model capacity ${MAX_MODEL_TYPES}`);
    return [];
  }

  const monomer = monomerLibrary.monomers[monomerIdx];
  console.log(`[generatePlacementsForMonomer] Generating ${numSamples} placements for ${monomer.code} (idx=${monomerIdx})`);

  // Encode target (reuse existing encoding)
  const encodedTarget = await encodeTargetMolecule();
  if (!encodedTarget) {
    console.error('[generatePlacementsForMonomer] Failed to encode target');
    return [];
  }

  const placements = [];
  const loadingMsg = document.querySelector('#loading p');
  const statusEl = $('#diffmip-status');

  // Generate samples one at a time (each call gives random centroids)
  // We need to override the monomer type selection
  for (let sampleIdx = 0; sampleIdx < numSamples; sampleIdx++) {
    // Update progress
    const progress = `Generating placements for ${monomer.code}: ${sampleIdx + 1}/${numSamples}...`;
    const placementProgress = (sampleIdx / numSamples) * 100;
    updateProgress('Generating Placements', progress, placementProgress);
    if (loadingMsg) loadingMsg.textContent = progress;
    if (statusEl) statusEl.text(progress).css('color', '#fbbf24');

    try {
      // Sample ONE centroid for ONE monomer type
      const centroidResult = await eulerSampleCentroidsForType(
        encodedTarget,
        monomerIdx,  // Specific monomer type
        50           // num_steps
      );

      if (!centroidResult) {
        console.warn(`[generatePlacementsForMonomer] Failed to sample centroid ${sampleIdx}`);
        continue;
      }

      const { centroid } = centroidResult;

      // Sample torsions for this centroid
      const dofResult = await eulerSampleTorsionsForCentroid(
        encodedTarget,
        centroid,
        monomerIdx,
        50  // num_steps
      );

      if (!dofResult) {
        console.warn(`[generatePlacementsForMonomer] Failed to sample torsions ${sampleIdx}`);
        continue;
      }

      const { euler, torsions } = dofResult;

      placements.push({
        code: `${monomer.code}_${sampleIdx}`,
        monomerIdx: monomerIdx,
        smiles: monomer.smiles,
        centroid: centroid,
        euler: euler,
        torsions: torsions
      });

    } catch (error) {
      console.error(`[generatePlacementsForMonomer] Error generating sample ${sampleIdx}:`, error);
    }
  }

  console.log(`[generatePlacementsForMonomer] Generated ${placements.length}/${numSamples} placements for ${monomer.code}`);
  return placements;
}

/**
 * Sample centroid for a SPECIFIC monomer type (recipe mode)
 */
async function eulerSampleCentroidsForType(encodedTarget, monomerIdx, numSteps) {
  const B = 1;  // Batch size
  const K = 1;  // Just one monomer

  // Create tensor with specific monomer type
  const monoTypes = new BigInt64Array([BigInt(monomerIdx)]);
  const monoTypesTensor = new ort.Tensor('int64', monoTypes, [B, K]);

  // Initialize random centroid
  let x_t = new Float32Array(B * K * 3);
  for (let i = 0; i < x_t.length; i++) {
    x_t[i] = randn();
  }

  // Euler integration
  const dt = 1.0 / numSteps;

  // Auto-detect model input naming convention (do this once before loop)
  const modelInputs = diffmipModels.centroidNet.inputNames || [];
  const usesTargetPrefix = modelInputs.includes('target_node_emb');

  for (let step = 0; step < numSteps; step++) {
    const t = step / numSteps;
    const tTensor = new ort.Tensor('float32', new Float32Array([t]), [B]);
    const centroidTensor = new ort.Tensor('float32', x_t, [B, K, 3]);

    // Use auto-detected naming convention (matching eulerSampleCentroids)
    const feeds = usesTargetPrefix ? {
      't': tTensor,
      'centroid_t': centroidTensor,
      'mono_type_idx': monoTypesTensor,
      'target_node_emb': encodedTarget.nodeEmb,
      'target_global_ctx': encodedTarget.globalCtx,
      'target_batch': encodedTarget.batch
    } : {
      't': tTensor,
      'centroid_t': centroidTensor,
      'mono_type_idx': monoTypesTensor,
      'node_emb': encodedTarget.nodeEmb,
      'global_ctx': encodedTarget.globalCtx,
      'batch': encodedTarget.batch
    };

    const results = await diffmipModels.centroidNet.run(feeds);
    const v_pred = results.velocity.data;

    // Euler step
    for (let i = 0; i < x_t.length; i++) {
      x_t[i] = x_t[i] + v_pred[i] * dt;
    }
  }

  return {
    centroid: [x_t[0], x_t[1], x_t[2]]
  };
}

/**
 * Sample torsions for a SPECIFIC centroid (recipe mode)
 */
async function eulerSampleTorsionsForCentroid(encodedTarget, centroid, monomerIdx, numSteps) {
  const B = 1;
  const K = 1;
  const monomer = monomerLibrary.monomers[monomerIdx];

  // Get DOF count - use maxDof from config (fixed model dimension)
  const numTorsions = monomer.torsion_quads?.length || 0;
  const D_dof = DIFFMIP_CONFIG.maxDof;  // Always use maxDof (20)

  // Create tensors
  const monoTypes = new BigInt64Array([BigInt(monomerIdx)]);
  const monoTypesTensor = new ort.Tensor('int64', monoTypes, [B, K]);

  const centroidTensor = new ort.Tensor('float32', new Float32Array(centroid), [B, K, 3]);

  // DOF mask: 1.0 for active DOFs (3 euler + numTorsions), 0.0 for padding
  const dofMask = new Float32Array(B * K * D_dof).fill(0.0);
  for (let i = 0; i < 3 + numTorsions; i++) {
    dofMask[i] = 1.0;
  }
  const dofMaskTensor = new ort.Tensor('float32', dofMask, [B, K, D_dof]);

  // Initialize random DOFs (only for active ones, rest are zero-padded)
  let x_t = new Float32Array(B * K * D_dof).fill(0.0);
  for (let i = 0; i < 3 + numTorsions; i++) {
    x_t[i] = randn();
  }

  // Euler integration
  const dt = 1.0 / numSteps;

  // Auto-detect model input names (do this once before loop)
  const modelInputs = diffmipModels.torsionNet.inputNames || [];
  const usesTargetPrefix = modelInputs.includes('target_node_emb');
  const usesSigma = modelInputs.includes('sigma');

  for (let step = 0; step < numSteps; step++) {
    const t = step / numSteps;
    const tTensor = new ort.Tensor('float32', new Float32Array([t]), [B]);
    const dofTensor = new ort.Tensor('float32', x_t, [B, K, D_dof]);

    // Use auto-detected naming convention (matching eulerSampleTorsions)
    const feeds = usesTargetPrefix ? {
      [usesSigma ? 'sigma' : 't']: tTensor,
      'dof_t': dofTensor,
      'dof_mask': dofMaskTensor,
      'centroid_0': centroidTensor,
      'mono_type_idx': monoTypesTensor,
      'target_node_emb': encodedTarget.nodeEmb,
      'target_global_ctx': encodedTarget.globalCtx,
      'target_batch': encodedTarget.batch
    } : {
      [usesSigma ? 'sigma' : 't']: tTensor,
      'dof_t': dofTensor,
      'dof_mask': dofMaskTensor,
      'centroid_0': centroidTensor,
      'mono_type_idx': monoTypesTensor,
      'node_emb': encodedTarget.nodeEmb,
      'global_ctx': encodedTarget.globalCtx,
      'batch': encodedTarget.batch
    };

    const results = await diffmipModels.torsionNet.run(feeds);
    const v_pred = results.velocity.data;

    // Euler step
    for (let i = 0; i < x_t.length; i++) {
      x_t[i] = x_t[i] + v_pred[i] * dt;
    }
  }

  // Return only the actual torsions (not the padding)
  return {
    euler: [x_t[0], x_t[1], x_t[2]],
    torsions: Array.from(x_t.slice(3, 3 + numTorsions))
  };
}

/**
 * Score a placement against the target
 */
async function scorePlacement(placement) {
  try {
    // Generate 3D structure from SMILES
    const result = generateStructureFromSMILES(placement.smiles);
    if (!result) {
      console.warn(`[scorePlacement] Failed to generate structure for ${placement.code}`);
      return null;
    }

    // Center coordinates
    const centroid = result.coords.reduce((acc, [x, y, z]) => [
      acc[0] + x / result.coords.length,
      acc[1] + y / result.coords.length,
      acc[2] + z / result.coords.length
    ], [0, 0, 0]);

    let centeredCoords = result.coords.map(([x, y, z]) => [
      x - centroid[0],
      y - centroid[1],
      z - centroid[2]
    ]);

    // Apply torsional angles if present
    if (!DISABLE_TORSION_APPLICATION && placement.torsions && placement.torsions.length > 0) {
      centeredCoords = applyTorsions(centeredCoords, result.bonds, result.numAtoms, placement.torsions);
    }

    // Apply Euler rotation
    const rotMatrix = eulerToRotationMatrix(placement.euler[0], placement.euler[1], placement.euler[2]);
    const transformedCoords = transformCoordinates(centeredCoords, rotMatrix, placement.centroid);

    // Create PDB from transformed coordinates
    const pdbLines = result.pdb.split('\n');
    const newPdbLines = pdbLines.map((line, lineIdx) => {
      const atomIdx = lineIdx - 4;
      if (atomIdx >= 0 && atomIdx < transformedCoords.length) {
        const [x, y, z] = transformedCoords[atomIdx];
        return `${x.toFixed(4).padStart(10, ' ')}${y.toFixed(4).padStart(10, ' ')}${z.toFixed(4).padStart(10, ' ')}${line.substring(30)}`;
      }
      return line;
    });

    const transformedPDB = newPdbLines.join('\n');

    // Score against target using physics
    if (typeof window.PhysicsScoring !== 'undefined' && rawPdbText) {
      const score = window.PhysicsScoring.scoreInteraction(rawPdbText, transformedPDB);

      // Store additional info in placement
      placement.positions = transformedCoords;
      placement.structureText = transformedPDB;

      return score;
    }

    return null;
  } catch (error) {
    console.error(`[scorePlacement] Error:`, error);
    return null;
  }
}

/**
 * Score a placement against an already-placed monomer
 */
async function scorePlacementAgainstPlaced(placement, placedMonomer) {
  try {
    if (!placement.structureText || !placedMonomer.structureText) {
      console.warn('[scorePlacementAgainstPlaced] Missing structure text');
      return null;
    }

    if (typeof window.PhysicsScoring !== 'undefined') {
      // Score interaction between two monomers
      const score = window.PhysicsScoring.scoreInteraction(placedMonomer.structureText, placement.structureText);
      return score;
    }

    return null;
  } catch (error) {
    console.error(`[scorePlacementAgainstPlaced] Error:`, error);
    return null;
  }
}

// =============================================================================
// MAIN INFERENCE ENTRY POINT
// =============================================================================

/**
 * Run complete diffMIP inference pipeline
 */
async function runDiffMIPClient() {
  console.log('[runDiffMIPClient] Starting inference...');

  // Check if inference is already running
  if (isInferenceRunning) {
    showToast('Inference already running. Please wait...');
    console.warn('[runDiffMIPClient] Inference already in progress');
    return;
  }

  // Check if clearing is in progress
  if (isClearing) {
    showToast('Cannot run inference while clearing placements');
    console.warn('[runDiffMIPClient] Clearing in progress');
    return;
  }

  // Clear any previous placements before starting new prediction
  console.log('[runDiffMIPClient] Clearing previous placements...');
  clearPlacements();

  // Check if models are loaded
  if (!diffmipModels || !monomerLibrary) {
    showToast('Models not loaded yet. Please wait or refresh the page.');
    console.error('[runDiffMIPClient] diffMIP models not loaded');
    return;
  }

  // Verify all model components exist
  if (!diffmipModels.encoder || !diffmipModels.centroidNet || !diffmipModels.torsionNet) {
    showToast('Model components missing. Please refresh the page.');
    console.error('diffMIP model components incomplete');
    return;
  }

  // Check if target molecule is loaded
  if (!rawPdbText) {
    showToast('Please load a molecule first');
    return;
  }

  const btn = $('#diffmip-predict');
  const mainBtn = $('#predict');
  const statusEl = $('#diffmip-status');

  // Set running flag
  isInferenceRunning = true;

  // Update both buttons
  btn.text('Running…').addClass('running').prop('disabled', true);
  mainBtn.text('Running…').addClass('running').prop('disabled', true);

  // Debug: Verify status element exists
  console.log('[runDiffMIPClient] Status element found:', statusEl.length > 0);
  if (statusEl.length > 0) {
    statusEl.text('Running inference...').css('color', '#fbbf24');
    console.log('[runDiffMIPClient] Status updated to: Running inference...');
  } else {
    console.error('[runDiffMIPClient] #diffmip-status element not found!');
  }

  // Status updates only (no loading modal)
  console.log('[runDiffMIPClient] Starting prediction...');

  // loadingMsg will be null since modal was removed, but keep for backwards compatibility
  const loadingMsg = null;

  try {
    // Get parameters
    const numMonomers = parseInt($('#diffmip-num-monomers').val()) || 5;
    const numSamples = parseInt($('#diffmip-num-samples').val()) || 1;
    const numSteps = parseInt($('#diffmip-num-steps').val()) || 50;

    console.log(`\nRunning client-side inference:`);
    console.log(`  Mode: ${USE_RECIPE_MODE ? 'RECIPE (iterative)' : 'SINGLE-SHOT'}`);
    console.log(`  Monomers: ${numMonomers}`);
    console.log(`  Samples: ${numSamples}`);
    console.log(`  Steps: ${numSteps}`);

    // RECIPE MODE: Iterative monomer placement
    if (USE_RECIPE_MODE) {
      console.log('\n🔁 Using RECIPE mode (iterative placement)');

      // Clear cached encoding for fresh start
      cachedEncodedTarget = null;

      const energyCutoff = parseFloat($('#diffmip-energy-cutoff').val()) || -3.0;

      const placedMonomers = await buildReceptorIteratively(
        numMonomers,      // maxMonomers
        numSamples,       // samplesPerMonomer
        5,                // topKCandidates
        energyCutoff      // energyThreshold
      );

      console.log(`\n✓ Recipe complete: ${placedMonomers.length} monomers placed`);

      // Convert to display format
      const placementsForDisplay = placedMonomers.map((placed, idx) => ({
        code: placed.code,
        smiles: placed.smiles,
        centroid: placed.centroid,
        euler: placed.euler,
        torsions: placed.torsions,
        score: placed.targetScore,
        structureText: placed.structureText
      }));

      // Compute scores and visualize (wrap in object with placements property)
      updateProgress('Finalizing Results', 'Visualizing placements...', 95);
      if (loadingMsg) loadingMsg.textContent = 'Visualizing results…';
      await showDiffMIPResults({ placements: placementsForDisplay });

      // Success
      statusEl.text(`✓ Recipe complete: ${placedMonomers.length} monomers`).css('color', '#4ade80');
      showToast(`Recipe complete: ${placedMonomers.length} monomers placed`);

      return;  // Exit early - recipe mode is complete
    }

    // SINGLE-SHOT MODE: Original approach (all monomers at once)
    console.log('\n⚡ Using SINGLE-SHOT mode (all at once)');

    // Parse PDB
    console.log('\n1. Parsing PDB...');
    updateProgress('Parsing Target', 'Parsing target molecule...', 5);
    if (loadingMsg) loadingMsg.textContent = 'Parsing target molecule…';
    const atoms = parsePDB(rawPdbText);
    console.log(`   ✓ Parsed ${atoms.length} atoms`);

    // Encode target
    console.log('\n2. Encoding target molecule...');
    updateProgress('Encoding Target', 'Encoding target molecule...', 10);
    if (loadingMsg) loadingMsg.textContent = 'Encoding target molecule…';
    const encodedTarget = await encodeTarget(atoms);
    console.log(`   ✓ Node embeddings: ${encodedTarget.nodeEmb.dims}`);
    console.log(`   ✓ Global context: ${encodedTarget.globalCtx.dims}`);

    // Generate samples
    const allPlacements = [];

    for (let sampleIdx = 0; sampleIdx < numSamples; sampleIdx++) {
      console.log(`\n3. Sample ${sampleIdx + 1}/${numSamples}:`);

      // Sample centroids
      console.log('   Sampling centroids...');
      const centroidProgress = 20 + (sampleIdx / numSamples) * 30;
      updateProgress('Sampling Centroids', `Sampling centroids (${sampleIdx + 1}/${numSamples})...`, centroidProgress);
      if (loadingMsg) loadingMsg.textContent = `Sampling centroids (${sampleIdx + 1}/${numSamples})…`;
      const { centroids, monoTypes } = await eulerSampleCentroids(
        encodedTarget,
        numMonomers,
        numSteps
      );
      console.log('   ✓ Centroids sampled');

      // Sample torsions
      console.log('   Sampling torsions...');
      const torsionProgress = 50 + (sampleIdx / numSamples) * 30;
      updateProgress('Sampling Torsions', `Sampling torsions (${sampleIdx + 1}/${numSamples})...`, torsionProgress);
      if (loadingMsg) loadingMsg.textContent = `Sampling torsions (${sampleIdx + 1}/${numSamples})…`;
      const dofs = await eulerSampleTorsions(
        encodedTarget,
        centroids,
        monoTypes,
        numMonomers,
        numSteps
      );
      console.log('   ✓ Torsions sampled');

      // Extract placements
      for (let k = 0; k < numMonomers; k++) {
        const monoIdx = Number(monoTypes[k]);

        // Validate index before accessing
        if (monoIdx < 0 || monoIdx >= monomerLibrary.monomers.length) {
          console.error(`[runDiffMIPClient] OUT OF BOUNDS: Trying to access monomer ${monoIdx}, but library only has ${monomerLibrary.monomers.length} monomers`);
          throw new Error(`Monomer index out of bounds: ${monoIdx} >= ${monomerLibrary.monomers.length}`);
        }

        const monomer = monomerLibrary.monomers[monoIdx];

        // Validate monomer data
        if (!monomer) {
          console.error(`[runDiffMIPClient] Monomer at index ${monoIdx} is undefined or null`);
          throw new Error(`Monomer at index ${monoIdx} is undefined`);
        }
        if (!monomer.smiles) {
          console.error(`[runDiffMIPClient] Monomer at index ${monoIdx} (${monomer.code}) has no SMILES`);
          throw new Error(`Monomer ${monomer.code} has no SMILES`);
        }

        // Validate centroid array access
        const centroidOffset = k * 3;
        if (centroidOffset + 3 > centroids.length) {
          console.error(`[runDiffMIPClient] OUT OF BOUNDS: Centroid offset ${centroidOffset} + 3 exceeds centroids.length ${centroids.length}`);
          throw new Error(`Centroid array access out of bounds for monomer ${k}`);
        }

        const centroid = [
          centroids[centroidOffset + 0],
          centroids[centroidOffset + 1],
          centroids[centroidOffset + 2]
        ];

        // First 3 DOFs are Euler angles, rest are torsion angles
        const dofOffset = k * DIFFMIP_CONFIG.maxDof;

        // Validate DOF array access
        if (dofOffset + DIFFMIP_CONFIG.maxDof > dofs.length) {
          console.error(`[runDiffMIPClient] OUT OF BOUNDS: DOF offset ${dofOffset} + ${DIFFMIP_CONFIG.maxDof} exceeds dofs.length ${dofs.length}`);
          throw new Error(`DOF array access out of bounds for monomer ${k}`);
        }

        const euler = [
          dofs[dofOffset + 0],
          dofs[dofOffset + 1],
          dofs[dofOffset + 2]
        ];

        // Extract torsion angles (remaining DOFs)
        const torsions = [];
        for (let d = 3; d < DIFFMIP_CONFIG.maxDof; d++) {
          torsions.push(dofs[dofOffset + d]);
        }

        allPlacements.push({
          sample: sampleIdx + 1,
          k: k + 1,
          code: monomer.code,
          fullname: monomer.fullname || monomer.code,
          smiles: monomer.smiles,
          centroid: centroid,
          euler: euler,
          torsions: torsions
        });
      }
    }

    // Prepare results
    const results = {
      placements: allPlacements,
      target: {
        num_atoms: atoms.length
      }
    };

    console.log(`\n✓ Generated ${allPlacements.length} placements`);

    // Show results (with scoring and filtering)
    updateProgress('Computing Scores', 'Computing scores and filtering results...', 85);
    if (loadingMsg) loadingMsg.textContent = 'Computing scores and filtering…';
    await showDiffMIPResults(results);

    statusEl.text('Ready (Client-Side)').css('color', '#4ade80');

  } catch (error) {
    console.error('Client-side inference error:', error);
    showToast('Error: ' + error.message);
    statusEl.text('Error - see console').css('color', '#f87171');
  } finally {
    // Clear running flag
    isInferenceRunning = false;

    // Restore both buttons
    btn.text('Generate Placements').removeClass('running').prop('disabled', false);
    mainBtn.text('Predict Placements').removeClass('running').prop('disabled', false);
  }
}


/**
 * Display diffMIP results and visualize in 3D viewer
 */
async function showDiffMIPResults(results) {
  const container = $('#diffmip-results');
  container.empty();

  if (!results.placements || results.placements.length === 0) {
    container.html('<div style="color: var(--text-muted); font-size: 0.85rem;">No placements generated</div>');
    container.show();
    return;
  }

  console.log(`[showDiffMIPResults] Computing scores for ${results.placements.length} placements...`);
  updateProgress('Computing Scores', `Computing scores for ${results.placements.length} placements...`, 90);

  // STEP 1: Compute scores for all placements (without adding to viewer)
  const scoredPlacements = await computeScoresForPlacements(results.placements);

  // Get energy cutoff from UI
  const inputValue = $('#diffmip-energy-cutoff').val();
  const energyCutoff = parseFloat(inputValue);
  console.log(`[showDiffMIPResults] Energy cutoff input value: "${inputValue}" → parsed: ${energyCutoff} kcal/mol`);

  // Use default if parsing failed
  const finalCutoff = isNaN(energyCutoff) ? -5 : energyCutoff;
  console.log(`[showDiffMIPResults] Using energy cutoff: ${finalCutoff} kcal/mol`);

  // STEP 2: Filter by energy cutoff (using energy_total to match Python diffMIP behavior)
  updateProgress('Filtering Results', `Filtering by energy cutoff (${finalCutoff.toFixed(1)} kcal/mol)...`, 92);
  const filteredPlacements = scoredPlacements.filter(p => {
    if (!p.score) return false; // Exclude if no score
    const passes = p.score.energy_total < finalCutoff;  // Use < not <= (matches Python)
    if (scoredPlacements.indexOf(p) < 3) { // Log first 3 for debugging
      console.log(`[showDiffMIPResults] ${p.code}: energy_total=${p.score.energy_total.toFixed(2)}, cutoff=${finalCutoff.toFixed(2)}, passes=${passes}`);
    }
    return passes;
  });

  console.log(`[showDiffMIPResults] ${filteredPlacements.length} of ${scoredPlacements.length} passed energy cutoff`);

  if (filteredPlacements.length === 0) {
    const numFiltered = scoredPlacements.length;
    container.html(`<div style="color: var(--text-muted); font-size: 0.85rem;">No placements passed energy cutoff (${finalCutoff.toFixed(1)} kcal/mol). All ${numFiltered} placements filtered out.</div>`);
    container.show();
    $('#clear-placements').show();
    showToast(`No placements passed energy cutoff of ${finalCutoff.toFixed(1)} kcal/mol`);
    return;
  }

  // STEP 3: Sort by score (lower = better binding, using energy_total)
  const sortedPlacements = filteredPlacements.slice().sort((a, b) => {
    return a.score.energy_total - b.score.energy_total;
  });

  // STEP 4: Take only top N best monomers (from user input)
  const numToShow = parseInt($('#diffmip-num-monomers').val()) || 5;
  const topPlacements = sortedPlacements.slice(0, numToShow);
  const bestScore = topPlacements[0].score.energy_total;
  const numFiltered = scoredPlacements.length - filteredPlacements.length;

  console.log(`[showDiffMIPResults] Showing top ${topPlacements.length} of ${filteredPlacements.length} placements (best: ${bestScore.toFixed(2)} kcal/mol)`);

  // STEP 5: Visualize ONLY the top N placements
  updateProgress('Visualizing Results', `Visualizing top ${topPlacements.length} placements...`, 95);
  await visualizeSelectedPlacements(topPlacements);

  // Store in global for reference
  currentPlacements = topPlacements;

  // Display top 5
  const resultsDiv = $('<div class="diffmip-sample">');
  resultsDiv.append(`<div class="diffmip-sample-header">Top ${topPlacements.length} Monomers (of ${filteredPlacements.length} passing cutoff)</div>`);

  topPlacements.forEach((p, idx) => {
    const row = $('<div class="diffmip-placement-row">');

    // Format score display (using energy_total for consistency)
    const energy = p.score.energy_total;
    const color = energy < -5.0 ? '#4ade80' : energy < -2.0 ? '#fbbf24' : '#f87171';
    const scoreHTML = `<span style="color: ${color}; font-weight: 600;">${energy.toFixed(2)}</span>`;

    // Add rank indicator
    const rank = `<span style="color: var(--text-muted); font-size: 0.85rem; margin-right: 0.5rem;">#${idx + 1}</span>`;

    row.html(`
      ${rank}
      <span class="diffmip-monomer-code">${p.code}</span>
      <span class="diffmip-monomer-name">${p.fullname || p.code}</span>
      <span class="diffmip-score">Score: ${scoreHTML}</span>
      <span class="diffmip-centroid">(${p.centroid.map(v => v.toFixed(1)).join(', ')})</span>
    `);

    // Find original index for highlighting
    const originalIdx = currentPlacements.indexOf(p);
    row.on('click', function() {
      highlightPlacement(p, originalIdx);
      $(this).addClass('selected').siblings().removeClass('selected');
    });

    resultsDiv.append(row);
  });

  container.append(resultsDiv);

  container.show();

  // Show clear button
  $('#clear-placements').show();

  // Final progress update
  updateProgress('Complete', `Successfully generated ${topPlacements.length} placements!`, 100);

  // Show success message
  showToast(`Showing top ${topPlacements.length} of ${filteredPlacements.length} placements (best: ${bestScore.toFixed(2)} kcal/mol, ${numFiltered} filtered out)`);
}


/**
 * Compute physics scores for placements without adding to viewer
 * This allows us to filter before visualizing
 *
 * Respects ENABLE_MONOMER_MONOMER_INTERACTIONS flag:
 * - true: Uses scoreMonomerInSystem (penalizes overlapping monomers)
 * - false: Uses scoreInteraction (target-monomer only, faster)
 */
async function computeScoresForPlacements(placements) {
  console.log(`[computeScoresForPlacements] Computing scores for ${placements.length} placements...`);

  // Check if RDKit is initialized
  if (!rdkitModule) {
    console.error('RDKit.js not initialized - cannot compute scores');
    return placements.map(p => ({ ...p, score: null, structureText: null }));
  }

  const loadingMsg = document.querySelector('#loading p');

  // PHASE 1: Generate all structures first (PARALLELIZED)
  console.log(`[computeScoresForPlacements] Phase 1: Generating structures in parallel...`);
  updateProgress('Generating Structures', `Generating 3D structures for ${placements.length} placements...`, 88);
  if (loadingMsg) {
    loadingMsg.textContent = 'Generating 3D structures…';
  }

  // Process all placements in parallel
  const structurePromises = placements.map(async (p, i) => {
    try {
      // Generate 3D structure from SMILES (synchronous operation)
      const result = generateStructureFromSMILES(p.smiles);

      if (!result) {
        console.warn(`Failed to generate structure for ${p.code}`);
        return { ...p, score: null, structureText: null };
      }

      // Center coordinates at origin
      const centroid = result.coords.reduce((acc, [x, y, z]) => [
        acc[0] + x / result.coords.length,
        acc[1] + y / result.coords.length,
        acc[2] + z / result.coords.length
      ], [0, 0, 0]);

      let centeredCoords = result.coords.map(([x, y, z]) => [
        x - centroid[0],
        y - centroid[1],
        z - centroid[2]
      ]);

      // APPLY TORSIONAL ANGLES (if present and enabled)
      if (!DISABLE_TORSION_APPLICATION && p.torsions && p.torsions.length > 0) {
        console.log(`[${p.code}] Applying ${p.torsions.length} torsional angles...`);
        try {
          centeredCoords = applyTorsions(centeredCoords, result.bonds, result.numAtoms, p.torsions);
          console.log(`[${p.code}] ✓ Torsions applied successfully`);
        } catch (e) {
          console.error(`[${p.code}] ERROR applying torsions:`, e);
          // Continue with non-torsioned coordinates on error
        }
      } else if (DISABLE_TORSION_APPLICATION) {
        console.log(`[${p.code}] Torsion application DISABLED (debug mode)`);
      } else {
        console.log(`[${p.code}] No torsional angles to apply (${p.torsions?.length || 0} torsions)`);
      }

      // Apply Euler rotation (rigid body)
      const rotMatrix = eulerToRotationMatrix(p.euler[0], p.euler[1], p.euler[2]);
      const transformedCoords = transformCoordinates(centeredCoords, rotMatrix, p.centroid);

      // Create PDB from transformed coordinates
      const pdbLines = result.pdb.split('\n');
      const newPdbLines = pdbLines.map((line, lineIdx) => {
        const atomIdx = lineIdx - 4;
        if (atomIdx >= 0 && atomIdx < transformedCoords.length) {
          const [x, y, z] = transformedCoords[atomIdx];
          return `${x.toFixed(4).padStart(10, ' ')}${y.toFixed(4).padStart(10, ' ')}${z.toFixed(4).padStart(10, ' ')}${line.substring(30)}`;
        }
        return line;
      });

      const transformedPDB = newPdbLines.join('\n');

      return {
        ...p,
        structureText: transformedPDB,
        positions: transformedCoords  // Store transformed coordinates for label positioning
      };

    } catch (error) {
      console.error(`Error processing ${p.code}:`, error);
      return { ...p, score: null, structureText: null };
    }
  });

  // Wait for all structures to be generated
  const placementsWithStructures = await Promise.all(structurePromises);

  console.log(`[computeScoresForPlacements] Phase 1 complete: Generated ${placementsWithStructures.length} structures (parallel)`);

  // PHASE 2: Score each placement (PARALLELIZED)
  if (ENABLE_MONOMER_MONOMER_INTERACTIONS) {
    console.log(`[computeScoresForPlacements] Phase 2: Computing scores WITH monomer-monomer interactions in parallel...`);
  } else {
    console.log(`[computeScoresForPlacements] Phase 2: Computing scores (target-monomer only) in parallel...`);
  }

  updateProgress('Computing Scores', `Scoring ${placementsWithStructures.length} placements...`, 92);
  if (loadingMsg && placementsWithStructures.length > 1) {
    loadingMsg.textContent = `Scoring placements…`;
  }

  // Process all scoring in parallel
  const scoringPromises = placementsWithStructures.map(async (p, i) => {
    if (!p.structureText) {
      return { ...p, score: null };
    }

    // Compute physics-based score
    let score = null;
    if (typeof window.PhysicsScoring !== 'undefined' && rawPdbText) {
      try {
        if (ENABLE_MONOMER_MONOMER_INTERACTIONS) {
          // Collect all OTHER monomer structures for this scoring
          const otherStructures = [];
          for (let j = 0; j < placementsWithStructures.length; j++) {
            if (i !== j && placementsWithStructures[j].structureText) {
              otherStructures.push(placementsWithStructures[j].structureText);
            }
          }

          // Use scoreMonomerInSystem which includes monomer-monomer interactions
          score = window.PhysicsScoring.scoreMonomerInSystem(rawPdbText, p.structureText, otherStructures);

          // Log if there are monomer-monomer clashes
          if (score && score.n_clashes_with_monomers > 0) {
            console.log(`[computeScoresForPlacements] ${p.code}: ${score.n_clashes_with_monomers} monomer-monomer clashes detected (energy: ${score.energy_monomer_monomer.toFixed(2)} kcal/mol)`);
          }
        } else {
          // Use scoreInteraction which only considers target-monomer interactions
          score = window.PhysicsScoring.scoreInteraction(rawPdbText, p.structureText);
        }

        // Log the computed score for first few placements
        if (i < 5 && score) {
          console.log(`[computeScoresForPlacements] ${p.code}: energy=${score.size_corrected_energy?.toFixed(2) || 'N/A'} kcal/mol (raw=${score.total_energy?.toFixed(2) || 'N/A'})`);
        }
      } catch (err) {
        console.warn(`Failed to score ${p.code}:`, err);
      }
    } else {
      console.warn(`Cannot score ${p.code}: PhysicsScoring=${typeof window.PhysicsScoring}, rawPdbText=${!!rawPdbText}`);
    }

    return {
      ...p,
      score: score
    };
  });

  // Wait for all scoring to complete
  const scoredPlacements = await Promise.all(scoringPromises);

  console.log(`[computeScoresForPlacements] ✓ Scored ${scoredPlacements.length} placements (parallel)`);

  // Log summary statistics
  const withScores = scoredPlacements.filter(p => p.score !== null);
  const withoutScores = scoredPlacements.filter(p => p.score === null);
  console.log(`[computeScoresForPlacements] Summary: ${withScores.length} with scores, ${withoutScores.length} without scores`);

  if (withScores.length > 0) {
    const energies = withScores.map(p => p.score.size_corrected_energy);
    const minEnergy = Math.min(...energies);
    const maxEnergy = Math.max(...energies);
    const avgEnergy = energies.reduce((a, b) => a + b, 0) / energies.length;
    console.log(`[computeScoresForPlacements] Energy range: ${minEnergy.toFixed(2)} to ${maxEnergy.toFixed(2)} kcal/mol (avg: ${avgEnergy.toFixed(2)})`);
  }

  // Log summary of clashes (only if monomer-monomer interactions are enabled)
  if (ENABLE_MONOMER_MONOMER_INTERACTIONS) {
    const totalClashes = scoredPlacements.reduce((sum, p) => {
      return sum + (p.score?.n_clashes_with_monomers || 0);
    }, 0);
    if (totalClashes > 0) {
      console.warn(`[computeScoresForPlacements] Total monomer-monomer clashes detected: ${totalClashes}`);
    }
  }

  return scoredPlacements;
}

/**
 * DEBUG: Helper function to check current placement stats
 * Call from browser console: checkPlacementStats()
 */
window.checkPlacementStats = function() {
  if (!currentPlacements || currentPlacements.length === 0) {
    console.log('No placements currently loaded. Run a prediction first.');
    return;
  }

  const withScores = currentPlacements.filter(p => p.score !== null);
  const withoutScores = currentPlacements.filter(p => p.score === null);

  console.log(`=== Placement Statistics ===`);
  console.log(`Total placements: ${currentPlacements.length}`);
  console.log(`With scores: ${withScores.length}`);
  console.log(`Without scores: ${withoutScores.length}`);

  if (withScores.length > 0) {
    const energies = withScores.map(p => p.score.size_corrected_energy);
    console.log(`Energy range: ${Math.min(...energies).toFixed(2)} to ${Math.max(...energies).toFixed(2)} kcal/mol`);
    console.log(`Average energy: ${(energies.reduce((a,b) => a+b, 0) / energies.length).toFixed(2)} kcal/mol`);

    console.log(`\nFirst 5 placements:`);
    withScores.slice(0, 5).forEach(p => {
      console.log(`  ${p.code}: ${p.score.size_corrected_energy.toFixed(2)} kcal/mol`);
    });
  }
};


/**
 * Visualize only selected placements in 3D viewer
 * Used after filtering to show only top N placements
 */
async function visualizeSelectedPlacements(placements) {
  console.log(`[visualizeSelectedPlacements] Visualizing ${placements.length} selected placements...`);

  // Check if viewer exists and is initialized
  if (typeof viewer === 'undefined' || !viewer) {
    console.error('3Dmol viewer not available or not fully initialized');
    return;
  }

  // Clear previous placement models
  placementShapes.forEach(model => {
    try {
      viewer.removeModel(model);
    } catch (e) {}
  });
  placementLabels.forEach(labelData => {
    try {
      if (labelData.label) {
        viewer.removeLabel(labelData.label);
      }
    } catch (e) {}
  });
  diffmipHBondShapes.forEach(shape => {
    try {
      viewer.removeShape(shape);
    } catch (e) {}
  });
  diffmipPiStackingShapes.forEach(shape => {
    try {
      viewer.removeShape(shape);
    } catch (e) {}
  });
  placementShapes = [];
  placementLabels = [];
  diffmipHBondShapes = [];
  diffmipPiStackingShapes = [];

  // Add each placement to viewer
  placements.forEach((p, idx) => {
    const color = getMonomerColor(idx);

    try {
      if (!p.structureText) {
        console.warn(`No structure for ${p.code}, using sphere`);
        addSpherePlacement(p, color);
        return;
      }

      // Add molecule model to viewer
      const model = viewer.addModel(p.structureText, 'sdf');

      // Get current radius settings from viewer.html (if available)
      const currentBondRadius = (typeof bondRadius !== 'undefined') ? bondRadius : 0.30;
      const currentAtomRadius = (typeof atomRadius !== 'undefined') ? atomRadius : 0.20;

      model.setStyle({}, {
        stick: {
          radius: currentBondRadius,
          color: color
        },
        sphere: {
          radius: currentAtomRadius,
          color: color
        }
      });
      placementShapes.push(model);

      // Add label (only if labels are visible)
      // Calculate centroid from actual transformed coordinates
      let labelPos;
      if (p.positions && p.positions.length > 0) {
        // Use transformed coordinates to get actual molecule position
        console.log(`[Label Debug] ${p.code}: Has ${p.positions.length} positions, first =`, p.positions[0]);
        const sumPos = p.positions.reduce((acc, pos) => [
          acc[0] + pos[0],
          acc[1] + pos[1],
          acc[2] + pos[2]
        ], [0, 0, 0]);
        const actualCentroid = sumPos.map(s => s / p.positions.length);
        console.log(`[Label Debug] ${p.code}: Calculated centroid =`, actualCentroid, 'vs placement.centroid =', p.centroid);
        labelPos = {
          x: actualCentroid[0],
          y: actualCentroid[1],
          z: actualCentroid[2] 
        };
      } else {
        // Fallback to placement centroid if positions not available
        console.warn(`[Label Debug] ${p.code}: No positions, using fallback centroid =`, p.centroid);
        labelPos = {
          x: p.centroid[0],
          y: p.centroid[1],
          z: p.centroid[2]
        };
      }

      // Check if labels should be visible (from viewer.html labelsVisible state)
      const shouldShowLabels = (typeof labelsVisible !== 'undefined' && labelsVisible);

      if (shouldShowLabels) {
        const label = viewer.addLabel(p.code, {
          position: labelPos,
          fontSize: 12,
          fontColor: 'white',
          backgroundColor: color,
          backgroundOpacity: 0.8,
          borderThickness: 1.5,
          borderColor: 'white',
          inFront: true
        });
        placementLabels.push({ label: label, pos: labelPos, code: p.code, color: color });
      } else {
        // Store label info for later if labels get toggled on
        placementLabels.push({ label: null, pos: labelPos, code: p.code, color: color });
      }

    } catch (error) {
      console.error(`Error visualizing ${p.code}:`, error);
      addSpherePlacement(p, color);
    }
  });

  viewer.render();
  console.log(`[visualizeSelectedPlacements] ✓ Visualized ${placements.length} placements`);

  // Visualize H-bonds and π-π stacking between target and placed monomers
  if (typeof rawPdbText !== 'undefined' && rawPdbText) {
    cachedTargetPdb = rawPdbText; // Cache for later redrawing with different opacity
    visualizeDiffMIPHBonds(rawPdbText, placements, 3.5);
    visualizePiStacking(rawPdbText, placements, 4.5, 30);
  } else {
    console.warn('[visualizeSelectedPlacements] No target PDB available for interaction visualization');
  }
}


// Store placement shapes and data for later manipulation
let placementShapes = [];
let placementLabels = [];
let diffmipHBondShapes = []; // H-bond visualization shapes - array of arrays, indexed by placement
let diffmipPiStackingShapes = []; // π-π stacking visualization shapes - array of arrays, indexed by placement
let currentPlacements = [];
let cachedTargetPdb = ''; // Cached target PDB for redrawing interactions
let isInferenceRunning = false; // Flag to prevent concurrent operations
let isClearing = false; // Flag to prevent operations during clear

/**
 * Show all diffMIP monomer labels
 */
function showDiffMIPLabels() {
  placementLabels.forEach(labelData => {
    if (!labelData.label && labelData.pos) {
      // Label was not shown, create it now
      const label = viewer.addLabel(labelData.code, {
        position: labelData.pos,
        fontSize: 12,
        fontColor: 'white',
        backgroundColor: labelData.color,
        backgroundOpacity: 0.8,
        borderThickness: 1.5,
        borderColor: 'white',
        inFront: true
      });
      labelData.label = label;
    }
  });
  viewer.render();
}

/**
 * Hide all diffMIP monomer labels
 */
function hideDiffMIPLabels() {
  placementLabels.forEach(labelData => {
    if (labelData.label) {
      try {
        viewer.removeLabel(labelData.label);
        labelData.label = null;
      } catch (e) {
        console.warn('Error removing label:', e);
      }
    }
  });
  viewer.render();
}

/**
 * Restore original assigned colors for diffMIP placed monomers
 * Called when "Rainbow" color mode is selected to restore each monomer's assigned color
 */
function restoreDiffMIPMonomerColors() {
  if (typeof viewer === 'undefined' || !viewer) return;
  if (placementShapes.length === 0 || placementLabels.length === 0) return;

  console.log('[restoreDiffMIPMonomerColors] Restoring colors for', placementShapes.length, 'monomers');

  // Get current style settings from viewer.html
  const currentBondRadius = (typeof bondRadius !== 'undefined') ? bondRadius : 0.30;
  const currentAtomRadius = (typeof atomRadius !== 'undefined') ? atomRadius : 0.20;

  // Restore each monomer to its original assigned color
  placementLabels.forEach((labelData, idx) => {
    if (idx < placementShapes.length) {
      const model = placementShapes[idx];
      const color = labelData.color;

      try {
        // Apply the original color and current radii to this specific model
        model.setStyle({}, {
          stick: {
            radius: currentBondRadius,
            color: color
          },
          sphere: {
            radius: currentAtomRadius,
            color: color
          }
        });
      } catch (e) {
        console.warn(`[restoreDiffMIPMonomerColors] Error restoring color for model ${idx}:`, e);
      }
    }
  });

  viewer.render();
  console.log('[restoreDiffMIPMonomerColors] ✓ Colors restored');
}

/**
 * Parse N and O heavy atoms from a PDB string for H-bond detection
 * @param {string} pdbText - PDB format text
 * @returns {Array} Array of {x, y, z, elem, resn} atom objects
 */
function parseHBondAtoms(pdbText) {
  const atoms = [];
  for (const line of pdbText.split('\n')) {
    const rec = line.substring(0, 6).trim();
    if (rec !== 'ATOM' && rec !== 'HETATM') continue;

    // Parse element (column 77-78, or extract from atom name)
    const rawElem = line.length > 76 ? line.substring(76, 78).trim() : '';
    const elem = (rawElem || line.substring(12, 16).trim().replace(/[^A-Za-z]/g, '')[0] || 'C')
                   .toUpperCase();

    // Only keep N and O atoms (potential H-bond donors/acceptors)
    if (elem !== 'N' && elem !== 'O') continue;

    atoms.push({
      x:    parseFloat(line.substring(30, 38)),
      y:    parseFloat(line.substring(38, 46)),
      z:    parseFloat(line.substring(46, 54)),
      resn: line.substring(17, 20).trim(),
      elem,
    });
  }
  return atoms;
}

/**
 * Parse N and O heavy atoms from an SDF/molblock string for H-bond detection
 * @param {string} sdfText - SDF/molblock format text
 * @returns {Array} Array of {x, y, z, elem} atom objects
 */
function parseHBondAtomsFromSDF(sdfText) {
  const atoms = [];
  const lines = sdfText.split('\n');

  // Line 4 (0-indexed line 3) contains atom/bond counts
  if (lines.length < 4) return atoms;

  const countsLine = lines[3];
  const atomCount = parseInt(countsLine.substring(0, 3).trim());

  if (isNaN(atomCount) || atomCount <= 0) return atoms;

  // Atom lines start at line 5 (0-indexed line 4)
  for (let i = 4; i < Math.min(4 + atomCount, lines.length); i++) {
    const line = lines[i];
    if (line.length < 34) continue;

    // SDF format: xxxxx.xxxxyyyyy.yyyyzzzzz.zzzz aaa
    // Coordinates are in columns 0-10, 10-20, 20-30
    // Element symbol is in columns 31-33
    const x = parseFloat(line.substring(0, 10).trim());
    const y = parseFloat(line.substring(10, 20).trim());
    const z = parseFloat(line.substring(20, 30).trim());
    const elem = line.substring(31, 34).trim().toUpperCase();

    // Validate coordinates
    if (isNaN(x) || isNaN(y) || isNaN(z)) continue;

    // Only keep N and O atoms (potential H-bond donors/acceptors)
    if (elem !== 'N' && elem !== 'O') continue;

    atoms.push({ x, y, z, elem });
  }

  return atoms;
}

/**
 * Draw a dashed line between two points to represent an H-bond
 * Uses thin cylinders alternating with gaps to create a dash pattern
 * @param {Object} p1 - Start point {x, y, z}
 * @param {Object} p2 - End point {x, y, z}
 * @param {number} opacity - Opacity of the dashes (0-1), default 1.0
 * @returns {Array} Array of 3Dmol shape references
 */
function drawHBondDash(p1, p2, opacity = 1.0) {
  const N_DASHES = 5;
  const dx = p2.x - p1.x, dy = p2.y - p1.y, dz = p2.z - p1.z;
  const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
  const ux = dx / len, uy = dy / len, uz = dz / len;
  const segLen = len / (2 * N_DASHES - 1);   // dash + gap unit
  const shapes = [];

  for (let i = 0; i < N_DASHES; i++) {
    const t0 = 2 * i * segLen;
    const t1 = (2 * i + 1) * segLen;
    shapes.push(viewer.addCylinder({
      start:   { x: p1.x + ux * t0, y: p1.y + uy * t0, z: p1.z + uz * t0 },
      end:     { x: p1.x + ux * t1, y: p1.y + uy * t1, z: p1.z + uz * t1 },
      radius:  0.07,
      color:   '#ffe066',   // warm yellow
      opacity: opacity,
      fromCap: 2,
      toCap:   2,
    }));
  }
  return shapes;
}

/**
 * Visualize hydrogen bonds between target and diffMIP placed monomers
 * @param {string} targetPdb - PDB text of target molecule
 * @param {Array} placements - Array of placement objects with structureText
 * @param {number} cutoff - Distance cutoff for H-bond detection (default 3.5 Å)
 * @param {number} highlightedIndex - Index of highlighted placement (-1 for none, full opacity for all)
 */
function visualizeDiffMIPHBonds(targetPdb, placements, cutoff = 5, highlightedIndex = -1) {
  if (!targetPdb || !placements || placements.length === 0) {
    console.log('[visualizeDiffMIPHBonds] No target or placements to visualize H-bonds');
    return;
  }

  // Clear previous H-bond shapes
  diffmipHBondShapes.forEach(shapesArray => {
    if (shapesArray) {
      shapesArray.forEach(s => {
        try {
          viewer.removeShape(s);
        } catch (e) {}
      });
    }
  });
  diffmipHBondShapes = [];

  // Parse N/O atoms from target
  const targetAtoms = parseHBondAtoms(targetPdb);
  console.log(`[visualizeDiffMIPHBonds] Target: ${targetAtoms.length} N/O atoms`);

  let totalHBonds = 0;

  // For each placement, find H-bonds with target
  placements.forEach((p, idx) => {
    if (!p.structureText) {
      diffmipHBondShapes[idx] = [];
      return;
    }

    // Initialize array for this placement's H-bonds
    diffmipHBondShapes[idx] = [];

    // Parse N/O atoms from monomer (SDF format)
    const monomerAtoms = parseHBondAtomsFromSDF(p.structureText);

    if (idx < 3) {  // Log first few monomers for debugging
      console.log(`[visualizeDiffMIPHBonds] Monomer ${idx} (${p.code}): ${monomerAtoms.length} N/O atoms`);
    }

    let hbondCount = 0;

    // Find all N/O pairs within cutoff distance
    for (const ta of targetAtoms) {
      for (const ma of monomerAtoms) {
        const dx = ta.x - ma.x, dy = ta.y - ma.y, dz = ta.z - ma.z;
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

        if (dist < cutoff) {
          // Determine opacity based on highlighted placement
          const opacity = (highlightedIndex === -1 || highlightedIndex === idx) ? 1.0 : 0.2;
          // Draw H-bond dash and store in this placement's array
          diffmipHBondShapes[idx].push(...drawHBondDash(ta, ma, opacity));
          hbondCount++;
          totalHBonds++;
        }
      }
    }

    if (hbondCount > 0) {
      console.log(`[visualizeDiffMIPHBonds] ${p.code}: ${hbondCount} H-bond(s) detected`);
    }
  });

  console.log(`[visualizeDiffMIPHBonds] ✓ Total ${totalHBonds} H-bond(s) detected at ${cutoff} Å cutoff`);

  // Show H-bond legend if any H-bonds were found
  if (totalHBonds > 0) {
    const legend = document.getElementById('hbond-legend');
    if (legend) {
      legend.classList.add('visible');
    }
  }

  viewer.render();
}

/**
 * Parse aromatic rings from PDB text
 * Detects common aromatic residues: PHE, TYR, TRP, HIS
 * @param {string} pdbText - PDB format text
 * @returns {Array} Array of {centroid: {x,y,z}, normal: {x,y,z}, atoms: [...]} ring objects
 */
function parseAromaticRingsFromPDB(pdbText) {
  const rings = [];
  const lines = pdbText.split('\n');

  // Define aromatic ring atoms for each residue type
  const aromaticRings = {
    PHE: ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],        // benzene ring
    TYR: ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],        // benzene ring
    TRP: ['CD2', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],      // benzene ring (indole)
    HIS: ['CG', 'ND1', 'CD2', 'CE1', 'NE2']               // imidazole ring
  };

  // Group atoms by residue
  const residueMap = {};
  for (const line of lines) {
    const rec = line.substring(0, 6).trim();
    if (rec !== 'ATOM' && rec !== 'HETATM') continue;

    const resn = line.substring(17, 20).trim();
    const resi = line.substring(22, 26).trim();
    const chain = line.substring(21, 22).trim();
    const atomName = line.substring(12, 16).trim();

    if (!aromaticRings[resn]) continue;

    const key = `${chain}:${resi}:${resn}`;
    if (!residueMap[key]) residueMap[key] = { resn, atoms: {} };

    residueMap[key].atoms[atomName] = {
      x: parseFloat(line.substring(30, 38)),
      y: parseFloat(line.substring(38, 46)),
      z: parseFloat(line.substring(46, 54))
    };
  }

  // Calculate ring centroids and normals
  for (const [key, res] of Object.entries(residueMap)) {
    const ringAtomNames = aromaticRings[res.resn];
    const ringAtoms = ringAtomNames
      .map(name => res.atoms[name])
      .filter(a => a && !isNaN(a.x) && !isNaN(a.y) && !isNaN(a.z));

    if (ringAtoms.length < 4) continue; // Need at least 4 atoms to define a ring

    // Calculate centroid
    const centroid = {
      x: ringAtoms.reduce((sum, a) => sum + a.x, 0) / ringAtoms.length,
      y: ringAtoms.reduce((sum, a) => sum + a.y, 0) / ringAtoms.length,
      z: ringAtoms.reduce((sum, a) => sum + a.z, 0) / ringAtoms.length
    };

    // Calculate normal vector using first 3 atoms (cross product)
    if (ringAtoms.length >= 3) {
      const v1 = {
        x: ringAtoms[1].x - ringAtoms[0].x,
        y: ringAtoms[1].y - ringAtoms[0].y,
        z: ringAtoms[1].z - ringAtoms[0].z
      };
      const v2 = {
        x: ringAtoms[2].x - ringAtoms[0].x,
        y: ringAtoms[2].y - ringAtoms[0].y,
        z: ringAtoms[2].z - ringAtoms[0].z
      };

      // Cross product for normal
      const normal = {
        x: v1.y * v2.z - v1.z * v2.y,
        y: v1.z * v2.x - v1.x * v2.z,
        z: v1.x * v2.y - v1.y * v2.x
      };

      // Normalize
      const len = Math.sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
      if (len > 0) {
        normal.x /= len;
        normal.y /= len;
        normal.z /= len;

        rings.push({ centroid, normal, atoms: ringAtoms });
      }
    }
  }

  return rings;
}

/**
 * Parse aromatic rings from SDF text
 * Uses bond information and atom types to detect rings
 * @param {string} sdfText - SDF/molblock format text
 * @returns {Array} Array of {centroid: {x,y,z}, normal: {x,y,z}, atoms: [...]} ring objects
 */
function parseAromaticRingsFromSDF(sdfText) {
  const rings = [];
  const lines = sdfText.split('\n');

  if (lines.length < 4) return rings;

  const countsLine = lines[3];
  const atomCount = parseInt(countsLine.substring(0, 3).trim());
  const bondCount = parseInt(countsLine.substring(3, 6).trim());

  if (isNaN(atomCount) || isNaN(bondCount)) return rings;

  // Parse atoms
  const atoms = [];
  for (let i = 4; i < Math.min(4 + atomCount, lines.length); i++) {
    const line = lines[i];
    if (line.length < 34) continue;

    atoms.push({
      x: parseFloat(line.substring(0, 10).trim()),
      y: parseFloat(line.substring(10, 20).trim()),
      z: parseFloat(line.substring(20, 30).trim()),
      elem: line.substring(31, 34).trim().toUpperCase()
    });
  }

  // Parse bonds to build adjacency list
  const bonds = [];
  const adjacency = Array.from({ length: atomCount }, () => []);
  for (let i = 4 + atomCount; i < Math.min(4 + atomCount + bondCount, lines.length); i++) {
    const line = lines[i];
    if (line.length < 9) continue;

    const atom1 = parseInt(line.substring(0, 3).trim()) - 1; // 1-indexed to 0-indexed
    const atom2 = parseInt(line.substring(3, 6).trim()) - 1;
    const bondType = parseInt(line.substring(6, 9).trim());

    if (!isNaN(atom1) && !isNaN(atom2) && atom1 >= 0 && atom2 >= 0 && atom1 < atomCount && atom2 < atomCount) {
      bonds.push({ atom1, atom2, bondType });
      adjacency[atom1].push(atom2);
      adjacency[atom2].push(atom1);
    }
  }

  // Simple ring detection: find 5- and 6-membered rings with aromatic atoms (C, N, O)
  const visited = new Set();

  for (let start = 0; start < atomCount; start++) {
    const elem = atoms[start].elem;
    if (elem !== 'C' && elem !== 'N' && elem !== 'O') continue;

    // DFS to find rings
    const findRings = (current, path, target, maxDepth) => {
      if (path.length > maxDepth) return;
      if (path.length >= 5 && adjacency[current].includes(target)) {
        // Found a ring
        const ringKey = [...path].sort().join(',');
        if (!visited.has(ringKey)) {
          visited.add(ringKey);
          const ringAtoms = path.map(idx => atoms[idx]);

          // Calculate centroid
          const centroid = {
            x: ringAtoms.reduce((sum, a) => sum + a.x, 0) / ringAtoms.length,
            y: ringAtoms.reduce((sum, a) => sum + a.y, 0) / ringAtoms.length,
            z: ringAtoms.reduce((sum, a) => sum + a.z, 0) / ringAtoms.length
          };

          // Calculate normal
          if (ringAtoms.length >= 3) {
            const v1 = {
              x: ringAtoms[1].x - ringAtoms[0].x,
              y: ringAtoms[1].y - ringAtoms[0].y,
              z: ringAtoms[1].z - ringAtoms[0].z
            };
            const v2 = {
              x: ringAtoms[2].x - ringAtoms[0].x,
              y: ringAtoms[2].y - ringAtoms[0].y,
              z: ringAtoms[2].z - ringAtoms[0].z
            };

            const normal = {
              x: v1.y * v2.z - v1.z * v2.y,
              y: v1.z * v2.x - v1.x * v2.z,
              z: v1.x * v2.y - v1.y * v2.x
            };

            const len = Math.sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
            if (len > 0) {
              normal.x /= len;
              normal.y /= len;
              normal.z /= len;

              rings.push({ centroid, normal, atoms: ringAtoms });
            }
          }
        }
      }

      for (const next of adjacency[current]) {
        if (next === target && path.length >= 5) continue; // Will be handled above
        if (path.includes(next)) continue;
        findRings(next, [...path, next], target, maxDepth);
      }
    };

    findRings(start, [start], start, 6);
  }

  return rings;
}

/**
 * Draw a double line between two ring centroids to represent π-π stacking
 * @param {Object} p1 - Start point {x, y, z}
 * @param {Object} p2 - End point {x, y, z}
 * @param {Object} normal1 - Normal vector of first ring
 * @param {number} opacity - Opacity of the lines (0-1), default 1.0
 * @returns {Array} Array of 3Dmol shape references
 */
function drawPiStackingLines(p1, p2, normal1, opacity = 1.0) {
  const shapes = [];

  // Create offset vector perpendicular to connection line
  const dx = p2.x - p1.x, dy = p2.y - p1.y, dz = p2.z - p1.z;
  const len = Math.sqrt(dx * dx + dy * dy + dz * dz);

  if (len < 0.01) return shapes;

  // Use ring normal to create offset
  const offsetDist = 0.15;
  const offset = {
    x: normal1.x * offsetDist,
    y: normal1.y * offsetDist,
    z: normal1.z * offsetDist
  };

  // Draw two parallel lines
  shapes.push(viewer.addCylinder({
    start: { x: p1.x + offset.x, y: p1.y + offset.y, z: p1.z + offset.z },
    end: { x: p2.x + offset.x, y: p2.y + offset.y, z: p2.z + offset.z },
    radius: 0.08,
    color: '#00ff88',  // cyan-green
    opacity: opacity,
    fromCap: 2,
    toCap: 2
  }));

  shapes.push(viewer.addCylinder({
    start: { x: p1.x - offset.x, y: p1.y - offset.y, z: p1.z - offset.z },
    end: { x: p2.x - offset.x, y: p2.y - offset.y, z: p2.z - offset.z },
    radius: 0.08,
    color: '#00ff88',  // cyan-green
    opacity: opacity,
    fromCap: 2,
    toCap: 2
  }));

  return shapes;
}

/**
 * Visualize π-π stacking interactions between target and diffMIP placed monomers
 * @param {string} targetPdb - PDB text of target molecule
 * @param {Array} placements - Array of placement objects with structureText
 * @param {number} distCutoff - Distance cutoff between ring centroids (default 4.5 Å)
 * @param {number} angleCutoff - Angle cutoff in degrees for parallel rings (default 30°)
 * @param {number} highlightedIndex - Index of highlighted placement (-1 for none, full opacity for all)
 */
function visualizePiStacking(targetPdb, placements, distCutoff = 4.5, angleCutoff = 30, highlightedIndex = -1) {
  if (!targetPdb || !placements || placements.length === 0) {
    console.log('[visualizePiStacking] No target or placements to visualize π-π stacking');
    return;
  }

  // Clear previous π-π stacking shapes
  if (typeof diffmipPiStackingShapes === 'undefined') {
    window.diffmipPiStackingShapes = [];
  }

  diffmipPiStackingShapes.forEach(shapesArray => {
    if (shapesArray) {
      shapesArray.forEach(s => {
        try {
          viewer.removeShape(s);
        } catch (e) {}
      });
    }
  });
  diffmipPiStackingShapes = [];

  // Parse aromatic rings from target
  const targetRings = parseAromaticRingsFromPDB(targetPdb);
  console.log(`[visualizePiStacking] Target: ${targetRings.length} aromatic ring(s)`);

  let totalStacking = 0;
  const angleThreshold = Math.cos(angleCutoff * Math.PI / 180);

  // For each placement, find π-π stacking with target
  placements.forEach((p, idx) => {
    if (!p.structureText) {
      diffmipPiStackingShapes[idx] = [];
      return;
    }

    // Initialize array for this placement's π-π stacking
    diffmipPiStackingShapes[idx] = [];

    // Parse aromatic rings from monomer (SDF format)
    const monomerRings = parseAromaticRingsFromSDF(p.structureText);

    if (idx < 3) {
      console.log(`[visualizePiStacking] Monomer ${idx} (${p.code}): ${monomerRings.length} aromatic ring(s)`);
    }

    let stackingCount = 0;

    // Check all ring pairs
    for (const tRing of targetRings) {
      for (const mRing of monomerRings) {
        // Distance check
        const dx = tRing.centroid.x - mRing.centroid.x;
        const dy = tRing.centroid.y - mRing.centroid.y;
        const dz = tRing.centroid.z - mRing.centroid.z;
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

        if (dist > distCutoff) continue;

        // Angle check - rings should be roughly parallel
        const dotProduct = Math.abs(
          tRing.normal.x * mRing.normal.x +
          tRing.normal.y * mRing.normal.y +
          tRing.normal.z * mRing.normal.z
        );

        if (dotProduct >= angleThreshold) {
          // Determine opacity based on highlighted placement
          const opacity = (highlightedIndex === -1 || highlightedIndex === idx) ? 1.0 : 0.2;
          // Draw π-π stacking indicator and store in this placement's array
          diffmipPiStackingShapes[idx].push(...drawPiStackingLines(tRing.centroid, mRing.centroid, tRing.normal, opacity));
          stackingCount++;
          totalStacking++;
        }
      }
    }

    if (stackingCount > 0) {
      console.log(`[visualizePiStacking] ${p.code}: ${stackingCount} π-π stacking interaction(s) detected`);
    }
  });

  console.log(`[visualizePiStacking] ✓ Total ${totalStacking} π-π stacking interaction(s) detected (dist ≤ ${distCutoff} Å, angle ≤ ${angleCutoff}°)`);

  // Show legend if any π-π stacking was found
  if (totalStacking > 0) {
    const legend = document.getElementById('pistacking-legend');
    if (legend) {
      legend.classList.add('visible');
    }
  }

  viewer.render();
}

/**
 * Generate color for monomer based on index
 */
function getMonomerColor(index) {
  const colors = [
    '#4ade80', // green
    '#60a5fa', // blue
    '#f472b6', // pink
    '#fbbf24', // yellow
    '#a78bfa', // purple
    '#fb923c', // orange
    '#2dd4bf', // teal
    '#f87171', // red
    '#94a3b8', // slate
    '#86efac', // light green
  ];
  return colors[index % colors.length];
}

/**
 * Convert Euler angles to rotation matrix (ZYX convention)
 */
function eulerToRotationMatrix(alpha, beta, gamma) {
  const ca = Math.cos(alpha), sa = Math.sin(alpha);
  const cb = Math.cos(beta), sb = Math.sin(beta);
  const cg = Math.cos(gamma), sg = Math.sin(gamma);

  // ZYX Euler angles rotation matrix
  return [
    [ca*cb, ca*sb*sg - sa*cg, ca*sb*cg + sa*sg],
    [sa*cb, sa*sb*sg + ca*cg, sa*sb*cg - ca*sg],
    [-sb, cb*sg, cb*cg]
  ];
}

/**
 * Apply rotation matrix and translation to coordinates
 */
function transformCoordinates(coords, rotMatrix, translation) {
  return coords.map(([x, y, z]) => {
    // Apply rotation
    const rx = rotMatrix[0][0]*x + rotMatrix[0][1]*y + rotMatrix[0][2]*z;
    const ry = rotMatrix[1][0]*x + rotMatrix[1][1]*y + rotMatrix[1][2]*z;
    const rz = rotMatrix[2][0]*x + rotMatrix[2][1]*y + rotMatrix[2][2]*z;

    // Apply translation
    return [
      rx + translation[0],
      ry + translation[1],
      rz + translation[2]
    ];
  });
}

/**
 * Rotate a vector around an axis using Rodrigues' formula
 */
function rotateVector(v, axis, angle) {
  // Normalize axis
  const axisLen = Math.sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]);
  const ax = [axis[0]/axisLen, axis[1]/axisLen, axis[2]/axisLen];

  // Rodrigues' rotation formula
  const cosA = Math.cos(angle);
  const sinA = Math.sin(angle);

  // v * cos(angle)
  const term1 = [v[0] * cosA, v[1] * cosA, v[2] * cosA];

  // (axis × v) * sin(angle)
  const cross = [
    ax[1]*v[2] - ax[2]*v[1],
    ax[2]*v[0] - ax[0]*v[2],
    ax[0]*v[1] - ax[1]*v[0]
  ];
  const term2 = [cross[0] * sinA, cross[1] * sinA, cross[2] * sinA];

  // axis * (axis · v) * (1 - cos(angle))
  const dot = ax[0]*v[0] + ax[1]*v[1] + ax[2]*v[2];
  const term3 = [ax[0] * dot * (1 - cosA), ax[1] * dot * (1 - cosA), ax[2] * dot * (1 - cosA)];

  return [
    term1[0] + term2[0] + term3[0],
    term1[1] + term2[1] + term3[1],
    term1[2] + term2[2] + term3[2]
  ];
}

/**
 * Apply a single torsional rotation around bond j-k
 */
function rotateBond(coords, bondJ, bondK, angle, atomsToRotate) {
  const newCoords = coords.map(c => [...c]); // Deep copy

  // Get bond axis
  const axis = [
    coords[bondK][0] - coords[bondJ][0],
    coords[bondK][1] - coords[bondJ][1],
    coords[bondK][2] - coords[bondJ][2]
  ];

  // Rotate each atom in atomsToRotate
  atomsToRotate.forEach(atomIdx => {
    // Translate to put bondJ at origin
    const translated = [
      coords[atomIdx][0] - coords[bondJ][0],
      coords[atomIdx][1] - coords[bondJ][1],
      coords[atomIdx][2] - coords[bondJ][2]
    ];

    // Rotate around axis
    const rotated = rotateVector(translated, axis, angle);

    // Translate back
    newCoords[atomIdx] = [
      rotated[0] + coords[bondJ][0],
      rotated[1] + coords[bondJ][1],
      rotated[2] + coords[bondJ][2]
    ];
  });

  return newCoords;
}

/**
 * Identify rotatable bonds from bond topology
 * Returns array of {bondIdx: i, atoms: [j, k]} for rotatable bonds
 */
function identifyRotatableBonds(bonds, numAtoms) {
  const rotatableBonds = [];

  bonds.forEach((bond, idx) => {
    const [atomJ, atomK] = bond.atoms;
    const bondOrder = bond.bo || 1;

    // Only single bonds can rotate freely
    if (bondOrder !== 1) return;

    // Build adjacency list to check if bond is in a ring
    // Simple heuristic: if both atoms have degree > 1, might be rotatable
    // (A full ring detection would require graph traversal)

    rotatableBonds.push({
      bondIdx: idx,
      atoms: [atomJ, atomK]
    });
  });

  return rotatableBonds;
}

/**
 * Build atom groups for each rotatable bond
 * Returns which atoms should rotate when the bond is rotated
 */
function buildAtomGroups(bonds, numAtoms, rotatableBond) {
  // Build adjacency list
  const adjacency = Array(numAtoms).fill(null).map(() => []);
  bonds.forEach(bond => {
    const [a, b] = bond.atoms;
    adjacency[a].push(b);
    adjacency[b].push(a);
  });

  // Find all atoms on the "k side" of bond j-k using BFS
  const [j, k] = rotatableBond.atoms;
  const visited = new Set();
  const queue = [k];
  visited.add(k);

  while (queue.length > 0) {
    const current = queue.shift();
    adjacency[current].forEach(neighbor => {
      // Don't cross back over the rotatable bond
      if (neighbor === j && current === k) return;

      if (!visited.has(neighbor)) {
        visited.add(neighbor);
        queue.push(neighbor);
      }
    });
  }

  // visited now contains all atoms on the k side (excluding j)
  // These are the atoms that should rotate
  return Array.from(visited);
}

/**
 * Apply torsional angles to molecular coordinates
 */
function applyTorsions(coords, bonds, numAtoms, torsionAngles) {
  if (!torsionAngles || torsionAngles.length === 0) {
    return coords;
  }

  // Identify rotatable bonds
  const rotatableBonds = identifyRotatableBonds(bonds, numAtoms);

  // Limit to number of provided torsion angles
  const numTorsions = Math.min(rotatableBonds.length, torsionAngles.length);

  if (numTorsions === 0) {
    return coords;
  }

  console.log(`[applyTorsions] Applying ${numTorsions} torsional angles to ${numAtoms} atoms`);
  console.log(`[applyTorsions] Rotatable bonds found: ${rotatableBonds.length}, torsion angles provided: ${torsionAngles.length}`);

  let currentCoords = coords.map(c => [...c]); // Deep copy

  // Apply each torsion sequentially
  for (let i = 0; i < numTorsions; i++) {
    const rotatableBond = rotatableBonds[i];
    const angle = torsionAngles[i];

    // Find which atoms to rotate
    const atomsToRotate = buildAtomGroups(bonds, numAtoms, rotatableBond);

    if (i < 3) {
      console.log(`[applyTorsions] Torsion ${i}: bond ${rotatableBond.atoms[0]}-${rotatableBond.atoms[1]}, angle=${angle.toFixed(3)} rad, rotating ${atomsToRotate.length} atoms`);
    }

    // Apply rotation
    currentCoords = rotateBond(
      currentCoords,
      rotatableBond.atoms[0],
      rotatableBond.atoms[1],
      angle,
      atomsToRotate
    );

    // Validate coordinates after rotation
    const hasNaN = currentCoords.some(c => isNaN(c[0]) || isNaN(c[1]) || isNaN(c[2]));
    if (hasNaN) {
      console.error(`[applyTorsions] NaN detected after applying torsion ${i}!`);
      throw new Error(`Invalid coordinates after torsion ${i}`);
    }
  }

  return currentCoords;
}

/**
 * Generate structure from SMILES using RDKit.js
 * Returns { pdb, coords, bonds, numAtoms } where bonds is topology for torsional application
 *
 * Note: RDKit.js MinimalLib doesn't have EmbedMolecule() or minimize() - those are Python-only.
 * We get 2D coordinates from get_molblock() and rely on torsional angles for 3D variation.
 */
function generateStructureFromSMILES(smiles) {
  if (!rdkitModule) {
    console.error('RDKit.js not initialized');
    return null;
  }

  try {
    // Create molecule from SMILES (without explicit hydrogens - united atom model)
    const mol = rdkitModule.get_mol(smiles);

    if (!mol || !mol.is_valid()) {
      console.error('Invalid SMILES:', smiles);
      return null;
    }

    // Extract bond topology using get_json() - module method is snake_case
    const jsonStr = mol.get_json();
    const molData = JSON.parse(jsonStr);

    const bonds = molData.molecules[0].bonds || [];
    const numAtoms = molData.molecules[0].atoms.length;

    // Get molblock - this provides 2D coordinates by default
    // (RDKit.js MinimalLib doesn't support 3D embedding)
    const pdb = mol.get_molblock();

    // Parse atom coordinates from molblock (SDF format)
    const lines = pdb.split('\n');
    const coords = [];

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];

      // Atom count is on line 4 (0-indexed line 3)
      if (i === 3) {
        const atomCount = parseInt(line.substring(0, 3).trim());
        if (atomCount !== numAtoms) {
          console.warn(`Atom count mismatch: molblock says ${atomCount}, JSON says ${numAtoms}`);
        }
        continue;
      }

      // Atom lines start at line 5 (0-indexed line 4)
      if (i >= 4 && i < 4 + numAtoms) {
        const x = parseFloat(line.substring(0, 10).trim());
        const y = parseFloat(line.substring(10, 20).trim());
        const z = parseFloat(line.substring(20, 30).trim());

        // Validate coordinates
        if (isNaN(x) || isNaN(y) || isNaN(z)) {
          console.error(`Invalid coordinates at atom ${i-4}: x=${x}, y=${y}, z=${z}`);
          mol.delete();
          return null;
        }

        coords.push([x, y, z]);
      }
    }

    if (coords.length === 0) {
      console.error('No coordinates found in molblock');
      mol.delete();
      return null;
    }

    // Check if coordinates are 2D or 3D
    const zValues = coords.map(c => c[2]);
    const hasVariableZ = zValues.some(z => Math.abs(z) > 0.01);

    if (!hasVariableZ) {
      // 2D coordinates - this is expected for RDKit.js
      // Torsional angles will add 3D variation
      console.log(`[generateStructureFromSMILES] Generated 2D structure: ${coords.length} atoms (torsions will add 3D variation)`);
    } else {
      const zRange = Math.max(...zValues) - Math.min(...zValues);
      console.log(`[generateStructureFromSMILES] Generated structure: ${coords.length} atoms, z-range: ${zRange.toFixed(2)} Å`);
    }

    console.log(`[generateStructureFromSMILES] Extracted ${bonds.length} bonds for ${numAtoms} atoms`);

    // Extract atom features and elements from JSON (RDKit MinimalLib has no get_atom method)
    const atomFeatures = [];
    const elements = [];

    // Extract elements from JSON atoms data
    const jsonAtoms = molData.molecules[0].atoms || [];
    for (let i = 0; i < numAtoms; i++) {
      // Get atomic number from JSON (label field contains element symbol)
      const atomData = jsonAtoms[i];
      const elementSymbol = atomData.label || 'C';

      // Map element symbol to atomic number
      const elementMap = {
        'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9,
        'P': 15, 'S': 16, 'Cl': 17, 'Br': 35
      };
      const atomicNum = elementMap[elementSymbol] || 6;
      elements.push(atomicNum);

      // Create atom features from JSON data (RDKit MinimalLib doesn't expose get_atom)
      const features = getAtomFeaturesFromJSON(atomData, atomicNum, bonds, i);
      atomFeatures.push(features);
    }

    console.log(`[generateStructureFromSMILES] Extracted ${numAtoms} atom features (24-dim each) from JSON`);

    // Compute torsion quads (rotatable bonds) from bond topology
    const torsionQuads = computeTorsionQuads(bonds, numAtoms);
    console.log(`[generateStructureFromSMILES] Computed ${torsionQuads.length} rotatable bonds`);

    return { pdb, coords, bonds, numAtoms, atomFeatures, elements, torsionQuads };

  } catch (error) {
    console.error('Error generating structure:', error);
    return null;
  }
}

/**
 * Compute torsion quads (i,j,k,l) for rotatable bonds
 * A rotatable bond is a single bond (not terminal) that's not in a small ring
 */
function computeTorsionQuads(bonds, numAtoms) {
  const torsionQuads = [];

  // Build adjacency list
  const adj = Array.from({ length: numAtoms }, () => []);
  for (const bond of bonds) {
    if (bond.atoms && bond.atoms.length === 2) {
      const [a, b] = bond.atoms;
      adj[a].push(b);
      adj[b].push(a);
    }
  }

  // For each bond, check if it's rotatable
  for (const bond of bonds) {
    if (!bond.atoms || bond.atoms.length !== 2) continue;

    const [j, k] = bond.atoms;

    // Skip if bond order is not 1 (if available)
    if (bond.bo && bond.bo !== 1) continue;

    // Need at least one neighbor on each side (not terminal)
    if (adj[j].length < 2 || adj[k].length < 2) continue;

    // Find neighbors for i and l
    const i_candidates = adj[j].filter(n => n !== k);
    const l_candidates = adj[k].filter(n => n !== j);

    if (i_candidates.length === 0 || l_candidates.length === 0) continue;

    // Take first neighbor on each side to form quad (i, j, k, l)
    const i = i_candidates[0];
    const l = l_candidates[0];

    torsionQuads.push([i, j, k, l]);
  }

  return torsionQuads;
}

/**
 * Visualize all placements in 3D viewer (client-side)
 */
async function visualizeAllPlacements(placements) {
  // Check if viewer exists and is initialized
  if (typeof viewer === 'undefined' || !viewer) {
    console.error('3Dmol viewer not available or not fully initialized');
    return;
  }

  // Check if RDKit is initialized
  if (!rdkitModule) {
    console.error('RDKit.js not initialized - falling back to spheres');
    visualizePlacementsAsSpheres(placements);
    return;
  }

  // Store placements globally for highlighting
  currentPlacements = placements;

  // Clear previous placement models
  placementShapes.forEach(model => {
    try {
      viewer.removeModel(model);
    } catch (e) {}
  });
  placementLabels.forEach(labelData => {
    try {
      if (labelData.label) {
        viewer.removeLabel(labelData.label);
      }
    } catch (e) {}
  });
  diffmipHBondShapes.forEach(shape => {
    try {
      viewer.removeShape(shape);
    } catch (e) {}
  });
  diffmipPiStackingShapes.forEach(shape => {
    try {
      viewer.removeShape(shape);
    } catch (e) {}
  });
  placementShapes = [];
  placementLabels = [];
  diffmipHBondShapes = [];
  diffmipPiStackingShapes = [];

  console.log(`Visualizing ${placements.length} placements...`);

  // Generate structures and compute physics scores
  const scoredPlacements = [];

  placements.forEach((p, idx) => {
    const color = getMonomerColor(idx);

    try {
      // Generate 3D structure from SMILES
      const result = generateStructureFromSMILES(p.smiles);

      if (!result) {
        console.warn(`Failed to generate structure for ${p.code}, using sphere`);
        addSpherePlacement(p, color);
        scoredPlacements.push({ ...p, score: null, structureText: null });
        return;
      }

      // Center coordinates at origin
      const centroid = result.coords.reduce((acc, [x, y, z]) => [
        acc[0] + x / result.coords.length,
        acc[1] + y / result.coords.length,
        acc[2] + z / result.coords.length
      ], [0, 0, 0]);

      const centeredCoords = result.coords.map(([x, y, z]) => [
        x - centroid[0],
        y - centroid[1],
        z - centroid[2]
      ]);

      // Apply Euler rotation
      const rotMatrix = eulerToRotationMatrix(p.euler[0], p.euler[1], p.euler[2]);
      const transformedCoords = transformCoordinates(centeredCoords, rotMatrix, p.centroid);

      // Create PDB from transformed coordinates
      const pdbLines = result.pdb.split('\n');
      const newPdbLines = pdbLines.map((line, lineIdx) => {
        // Update atom coordinate lines
        const atomIdx = lineIdx - 4;
        if (atomIdx >= 0 && atomIdx < transformedCoords.length) {
          const [x, y, z] = transformedCoords[atomIdx];
          // Replace coordinates in molblock format (positions 0-30)
          return `${x.toFixed(4).padStart(10, ' ')}${y.toFixed(4).padStart(10, ' ')}${z.toFixed(4).padStart(10, ' ')}${line.substring(30)}`;
        }
        return line;
      });

      const transformedPDB = newPdbLines.join('\n');

      // Compute physics-based score
      let score = null;
      if (typeof window.PhysicsScoring !== 'undefined' && rawPdbText) {
        try {
          score = window.PhysicsScoring.scoreInteraction(rawPdbText, transformedPDB);
          console.log(`Score for ${p.code}:`, score?.size_corrected_energy?.toFixed(2));
        } catch (err) {
          console.warn(`Failed to score ${p.code}:`, err);
        }
      }

      // Store placement with score and structure
      scoredPlacements.push({
        ...p,
        score: score,
        structureText: transformedPDB,
        color: color
      });

      // Add molecule model to viewer
      const model = viewer.addModel(transformedPDB, 'sdf');

      // Get current radius settings from viewer.html (if available)
      const currentBondRadius = (typeof bondRadius !== 'undefined') ? bondRadius : 0.30;
      const currentAtomRadius = (typeof atomRadius !== 'undefined') ? atomRadius : 0.20;

      model.setStyle({}, {
        stick: {
          radius: currentBondRadius,
          color: color
        },
        sphere: {
          radius: currentAtomRadius,
          color: color
        }
      });
      placementShapes.push(model);

      // Add label at actual transformed position
      // Calculate centroid from transformed coordinates
      const sumPos = transformedCoords.reduce((acc, pos) => [
        acc[0] + pos[0],
        acc[1] + pos[1],
        acc[2] + pos[2]
      ], [0, 0, 0]);
      const actualCentroid = sumPos.map(s => s / transformedCoords.length);
      const labelPos = {
        x: actualCentroid[0],
        y: actualCentroid[1],
        z: actualCentroid[2] + 2.0  // Offset above molecule
      };

      const label = viewer.addLabel(p.code, {
        position: labelPos,
        fontSize: 12,
        fontColor: 'white',
        backgroundColor: color,
        backgroundOpacity: 0.8,
        borderThickness: 1.5,
        borderColor: 'white',
        inFront: true
      });
      placementLabels.push(label);

    } catch (error) {
      console.error(`Error visualizing ${p.code}:`, error);
      addSpherePlacement(p, color);
      scoredPlacements.push({ ...p, score: null, structureText: null });
    }
  });

  // Update placements with scores
  currentPlacements = scoredPlacements;

  viewer.render();
  console.log(`✓ Visualized ${placements.length} molecular structures (client-side)`);
}

/**
 * Fallback: visualize placements as spheres
 */
function visualizePlacementsAsSpheres(placements) {
  if (typeof viewer === 'undefined' || !viewer) return;

  currentPlacements = placements;
  diffmipHBondShapes.forEach(shape => {
    try {
      viewer.removeShape(shape);
    } catch (e) {}
  });
  diffmipPiStackingShapes.forEach(shape => {
    try {
      viewer.removeShape(shape);
    } catch (e) {}
  });
  placementShapes = [];
  placementLabels = [];
  diffmipHBondShapes = [];
  diffmipPiStackingShapes = [];

  placements.forEach((p, idx) => {
    const color = getMonomerColor(idx);
    addSpherePlacement(p, color);
  });

  viewer.render();
  console.log(`✓ Visualized ${placements.length} placements (spheres)`);
}

/**
 * Helper: add a single sphere placement
 */
function addSpherePlacement(placement, color) {
  const center = {
    x: placement.centroid[0],
    y: placement.centroid[1],
    z: placement.centroid[2]
  };

  const sphere = viewer.addSphere({
    center: center,
    radius: 1.5,
    color: color,
    alpha: 0.7
  });
  placementShapes.push(sphere);

  const label = viewer.addLabel(placement.code, {
    position: center,
    fontSize: 12,
    fontColor: 'white',
    backgroundColor: color,
    backgroundOpacity: 0.8,
    borderThickness: 1.5,
    borderColor: 'white',
    inFront: true
  });
  placementLabels.push(label);
}

/**
 * Update interaction shape opacities based on highlighted placement
 * @param {number} highlightedIndex - Index of the highlighted placement (-1 for none)
 */
function updateInteractionOpacity(highlightedIndex) {
  if (!cachedTargetPdb || currentPlacements.length === 0) {
    console.warn('[updateInteractionOpacity] No cached data available');
    return;
  }

  // Redraw H-bonds with updated opacity
  visualizeDiffMIPHBonds(cachedTargetPdb, currentPlacements, 3.5, highlightedIndex);

  // Redraw π-π stacking with updated opacity
  visualizePiStacking(cachedTargetPdb, currentPlacements, 4.5, 30, highlightedIndex);
}

/**
 * Highlight a specific placement
 */
function highlightPlacement(placement, index) {
  if (typeof viewer === 'undefined' || !viewer) {
    console.error('3Dmol viewer not available or not fully initialized');
    return;
  }

  if (currentPlacements.length === 0 || placementShapes.length === 0) {
    console.error('No placements to highlight');
    return;
  }

  // Update styles for all placement models
  placementShapes.forEach((model, idx) => {
    const isSelected = idx === index;
    const color = getMonomerColor(idx);

    // Highlighted placement has different style
    model.setStyle({}, {
      stick: {
        radius: isSelected ? 0.30 : 0.2, // bondRadius of selected monomer on click
        color: color,
        opacity: isSelected ? 1.0 : 0.5
      }
    });
  });

  // Update interaction shape opacities
  updateInteractionOpacity(index);

  viewer.render();

  console.log(`Highlighted: ${placement.code} at (${placement.centroid.join(', ')})`);
}

/**
 * Clear all placement visualizations
 */
function clearPlacements() {
  console.log('[clearPlacements] Starting clear operation...');

  try {
    // Set clearing flag to block any other operations
    isClearing = true;

    // Don't allow clearing while inference is running
    if (isInferenceRunning) {
      showToast('Cannot clear while inference is running');
      console.warn('[clearPlacements] Blocked: inference in progress');
      isClearing = false;
      return;
    }

    if (typeof viewer === 'undefined' || !viewer) {
      console.warn('[clearPlacements] 3Dmol viewer not available');
      isClearing = false;
      return;
    }

    console.log('[clearPlacements] Removing', placementShapes.length, 'models');

    // Remove all placement models
    placementShapes.forEach((model, idx) => {
      try {
        console.log(`[clearPlacements] Removing model ${idx + 1}/${placementShapes.length}`);
        viewer.removeModel(model);
      } catch (e) {
        console.warn(`[clearPlacements] Error removing model ${idx}:`, e.message);
      }
    });

    console.log('[clearPlacements] Removing', placementLabels.length, 'labels');

    // Remove all labels
    placementLabels.forEach((labelData, idx) => {
      try {
        console.log(`[clearPlacements] Removing label ${idx + 1}/${placementLabels.length}`);
        if (labelData.label) {
          viewer.removeLabel(labelData.label);
        }
      } catch (e) {
        console.warn(`[clearPlacements] Error removing label ${idx}:`, e.message);
      }
    });

    // Count total H-bond shapes
    let totalHBondShapes = 0;
    diffmipHBondShapes.forEach(shapesArray => {
      if (shapesArray) totalHBondShapes += shapesArray.length;
    });
    console.log('[clearPlacements] Removing', totalHBondShapes, 'H-bond shapes');

    // Remove all H-bond shapes (nested arrays)
    diffmipHBondShapes.forEach((shapesArray, placementIdx) => {
      if (shapesArray) {
        shapesArray.forEach((shape, shapeIdx) => {
          try {
            viewer.removeShape(shape);
          } catch (e) {
            console.warn(`[clearPlacements] Error removing H-bond shape [${placementIdx}][${shapeIdx}]:`, e.message);
          }
        });
      }
    });

    // Count total π-π stacking shapes
    let totalPiStackingShapes = 0;
    diffmipPiStackingShapes.forEach(shapesArray => {
      if (shapesArray) totalPiStackingShapes += shapesArray.length;
    });
    console.log('[clearPlacements] Removing', totalPiStackingShapes, 'π-π stacking shapes');

    // Remove all π-π stacking shapes (nested arrays)
    diffmipPiStackingShapes.forEach((shapesArray, placementIdx) => {
      if (shapesArray) {
        shapesArray.forEach((shape, shapeIdx) => {
          try {
            viewer.removeShape(shape);
          } catch (e) {
            console.warn(`[clearPlacements] Error removing π-π stacking shape [${placementIdx}][${shapeIdx}]:`, e.message);
          }
        });
      }
    });

    // Hide interaction legends
    const hbondLegend = document.getElementById('hbond-legend');
    if (hbondLegend) {
      hbondLegend.classList.remove('visible');
    }
    const piStackingLegend = document.getElementById('pistacking-legend');
    if (piStackingLegend) {
      piStackingLegend.classList.remove('visible');
    }

    // Clear arrays
    console.log('[clearPlacements] Clearing placement arrays');
    placementShapes = [];
    placementLabels = [];
    diffmipHBondShapes = [];
    diffmipPiStackingShapes = [];
    currentPlacements = [];
    cachedTargetPdb = '';

    // Safely render
    try {
      console.log('[clearPlacements] Rendering viewer');
      viewer.render();
    } catch (e) {
      console.warn('[clearPlacements] Error rendering after clear:', e.message);
    }

    // Clear results display
    try {
      console.log('[clearPlacements] Clearing UI elements');
      $('#diffmip-results').hide().empty();
      $('#clear-placements').hide();
      $('.diffmip-placement-row').removeClass('selected');
    } catch (e) {
      console.warn('[clearPlacements] Error clearing UI:', e.message);
    }

    console.log('[clearPlacements] ✓ Cleared all placements successfully');

  } catch (error) {
    console.error('[clearPlacements] ERROR:', error);
    console.error('[clearPlacements] Error stack:', error.stack);
    console.error('[clearPlacements] This error is NOT related to ONNX models - it is a viewer cleanup issue');
    showToast('Error clearing placements - see console');
  } finally {
    // Always clear the flag, even if there was an error
    console.log('[clearPlacements] Clearing isClearing flag');
    isClearing = false;
  }
}

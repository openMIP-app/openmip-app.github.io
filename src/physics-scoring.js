/**
 * Client-Side Physics-Based Scoring for Molecular Placements
 *
 * Implementation matching diffmip-lite's physics scoring approach.
 * Computes interaction energies using classical force field terms:
 *   - Van der Waals (Lennard-Jones 12-6)
 *   - Electrostatics (Coulomb with distance-dependent dielectric)
 *   - Hydrogen bonds (geometric criterion)
 *   - Steric clashes (quadratic penalty)
 *
 * Uses united atom model parameters (heavy atoms with implicit hydrogens).
 * Energies are clipped to prevent numerical overflow from close contacts.
 *
 * Based on diffmip-lite/utils/physics_scoring.py
 */

// Lennard-Jones parameters (UFF-like) - United Atom model
const LJ_SIGMA = {
  1: 2.571,  // H
  6: 3.50,   // C (with implicit H)
  7: 3.25,   // N (with implicit H)
  8: 3.07,   // O (with implicit H)
  9: 3.364,  // F
  15: 3.74,  // P
  16: 3.55,  // S
  17: 3.947, // Cl
  35: 4.189, // Br
};

const LJ_EPSILON = {
  1: 0.044,  // H
  6: 0.066,  // C
  7: 0.056,  // N
  8: 0.050,  // O
  9: 0.050,  // F
  15: 0.200, // P
  16: 0.175, // S
  17: 0.227, // Cl
  35: 0.251, // Br
};

/**
 * Parse atomic data from PDB text
 */
function parseAtomsFromPDB(pdbText) {
  const atoms = [];
  const lines = pdbText.split('\n');

  for (const line of lines) {
    if (line.startsWith('ATOM') || line.startsWith('HETATM')) {
      const x = parseFloat(line.substring(30, 38).trim());
      const y = parseFloat(line.substring(38, 46).trim());
      const z = parseFloat(line.substring(46, 54).trim());
      const element = line.substring(76, 78).trim() || line.substring(12, 16).trim()[0];

      // Map element symbol to atomic number
      const elementMap = {
        'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9,
        'P': 15, 'S': 16, 'Cl': 17, 'Br': 35
      };

      const atomicNum = elementMap[element] || 6; // default to carbon

      atoms.push({
        x, y, z,
        atomicNum,
        element
      });
    }
  }

  return atoms;
}

/**
 * Parse atomic data from SDF/molblock format
 */
function parseAtomsFromSDF(sdfText) {
  const atoms = [];
  const lines = sdfText.split('\n');

  // Atom count is on line 4 (0-indexed line 3)
  if (lines.length < 4) return atoms;

  const atomCount = parseInt(lines[3].substring(0, 3).trim());

  // Atom lines start at line 5 (0-indexed line 4)
  for (let i = 4; i < 4 + atomCount && i < lines.length; i++) {
    const line = lines[i];
    const x = parseFloat(line.substring(0, 10).trim());
    const y = parseFloat(line.substring(10, 20).trim());
    const z = parseFloat(line.substring(20, 30).trim());
    const element = line.substring(31, 34).trim();

    // Map element symbol to atomic number
    const elementMap = {
      'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9,
      'P': 15, 'S': 16, 'Cl': 17, 'Br': 35
    };

    const atomicNum = elementMap[element] || 6;

    atoms.push({
      x, y, z,
      atomicNum,
      element
    });
  }

  return atoms;
}

/**
 * Estimate partial charges using chemistry-based values (matching diffmip-lite)
 *
 * For united atom models (heavy atoms with implicit hydrogens):
 * - Charges are reduced because bonded H's partially neutralize heavy atoms
 * - Example: CH3 group has smaller effective charge than bare C
 */
function estimatePartialCharges(atoms) {
  const charges = atoms.map(atom => {
    const z = atom.atomicNum;

    // United atom charges - heavy atoms with implicit hydrogens
    // Based on diffmip-lite's physics_scoring.py
    if (z === 1) return 0.1;     // H (rarely present in united atom)
    if (z === 6) return 0.0;     // C with implicit H (CH, CH2, CH3) - nearly neutral
    if (z === 7) return -0.05;   // N with implicit H (NH, NH2, NH3) - slightly negative
    if (z === 8) return -0.10;   // O with implicit H (OH) - negative
    if (z === 9) return -0.20;   // F (no H typically) - very negative
    if (z === 15) return 0.0;    // P with implicit H - neutral
    if (z === 16) return -0.05;  // S with implicit H (SH) - slightly negative
    if (z === 17) return -0.10;  // Cl (no H) - negative
    if (z === 35) return -0.05;  // Br (no H) - slightly negative

    return 0.0;  // Default for unknown elements
  });

  // Normalize charges to ensure zero net charge (only if multiple atoms)
  // This is important for physical correctness
  if (charges.length > 1) {
    const mean = charges.reduce((sum, q) => sum + q, 0) / charges.length;
    return charges.map(q => q - mean);
  }

  return charges;
}

/**
 * Compute Van der Waals energy using Lennard-Jones potential
 */
function computeVdWEnergy(atomsA, atomsB, cutoff = 8.0) {
  let energy = 0.0;

  for (const a of atomsA) {
    const sigma_a = LJ_SIGMA[a.atomicNum] || 3.5;
    const eps_a = LJ_EPSILON[a.atomicNum] || 0.066;

    for (const b of atomsB) {
      const dx = a.x - b.x;
      const dy = a.y - b.y;
      const dz = a.z - b.z;
      const r = Math.sqrt(dx*dx + dy*dy + dz*dz);

      if (r > cutoff || r < 0.1) continue;

      const sigma_b = LJ_SIGMA[b.atomicNum] || 3.5;
      const eps_b = LJ_EPSILON[b.atomicNum] || 0.066;

      // Lorentz-Berthelot combining rules
      const sigma = (sigma_a + sigma_b) / 2.0;
      const epsilon = Math.sqrt(eps_a * eps_b);

      // Lennard-Jones 12-6 potential
      const sr = sigma / r;
      const sr6 = sr ** 6;
      const sr12 = sr6 * sr6;

      const E_lj = 4.0 * epsilon * (sr12 - sr6);

      // Clip extreme values to prevent numerical overflow (matching diffmip-lite)
      const E_lj_clipped = Math.max(-10.0, Math.min(100.0, E_lj));
      energy += E_lj_clipped;
    }
  }

  return energy;
}

/**
 * Compute electrostatic energy using Coulomb potential
 */
function computeElectrostaticEnergy(atomsA, chargesA, atomsB, chargesB, cutoff = 12.0) {
  let energy = 0.0;
  const k = 332.0636;  // Coulomb constant (kcal*Å/(mol*e²))

  for (let i = 0; i < atomsA.length; i++) {
    const a = atomsA[i];
    const qa = chargesA[i];

    for (let j = 0; j < atomsB.length; j++) {
      const b = atomsB[j];
      const qb = chargesB[j];

      const dx = a.x - b.x;
      const dy = a.y - b.y;
      const dz = a.z - b.z;
      const r = Math.sqrt(dx*dx + dy*dy + dz*dz);

      if (r > cutoff || r < 0.1) continue;

      // Distance-dependent dielectric: ε(r) = 4r
      const epsilon_r = 4.0 * r;

      const E_coulomb = k * qa * qb / (epsilon_r * r);

      // Clip extreme values to prevent numerical overflow (matching diffmip-lite)
      const E_coulomb_clipped = Math.max(-20.0, Math.min(20.0, E_coulomb));
      energy += E_coulomb_clipped;
    }
  }

  return energy;
}

/**
 * Compute hydrogen bond energy
 */
function computeHBondEnergy(atomsA, atomsB, cutoff = 3.5) {
  let energy = 0.0;
  let count = 0;

  // Identify potential H-bond donors and acceptors
  const donors_a = atomsA.filter(a => a.atomicNum === 7 || a.atomicNum === 8); // N, O
  const acceptors_b = atomsB.filter(b => b.atomicNum === 7 || b.atomicNum === 8);

  for (const donor of donors_a) {
    for (const acceptor of acceptors_b) {
      const dx = donor.x - acceptor.x;
      const dy = donor.y - acceptor.y;
      const dz = donor.z - acceptor.z;
      const r = Math.sqrt(dx*dx + dy*dy + dz*dz);

      if (r <= cutoff && r > 0.1) {
        // Simple H-bond potential: -2.0 kcal/mol at optimal distance
        const r0 = 2.8;  // Optimal H-bond distance (Å)
        const hbond = -2.0 * Math.exp(-((r - r0)**2) / 0.5);
        energy += hbond;
        count++;
      }
    }
  }

  return { energy, count };
}

/**
 * Compute steric clash penalty
 */
function computeClashPenalty(atomsA, atomsB, threshold = 2.0) {
  let penalty = 0.0;
  let count = 0;

  for (const a of atomsA) {
    for (const b of atomsB) {
      const dx = a.x - b.x;
      const dy = a.y - b.y;
      const dz = a.z - b.z;
      const r = Math.sqrt(dx*dx + dy*dy + dz*dz);

      if (r < threshold && r > 0.01) {
        // Harsh penalty for steric clashes
        penalty += 10.0 * (threshold - r) ** 2;
        count++;
      }
    }
  }

  return { penalty, count };
}

/**
 * Score interaction between target and monomer
 */
function scoreInteraction(targetPDB, monomerStructure, options = {}) {
  const {
    cutoff_vdw = 8.0,
    cutoff_elec = 12.0,
    cutoff_hbond = 3.5,
    clash_threshold = 2.0
  } = options;

  // Parse atoms
  const atomsTarget = parseAtomsFromPDB(targetPDB);
  const atomsMonomer = parseAtomsFromSDF(monomerStructure) || parseAtomsFromPDB(monomerStructure);

  if (atomsTarget.length === 0 || atomsMonomer.length === 0) {
    return null;
  }

  // Estimate partial charges
  const chargesTarget = estimatePartialCharges(atomsTarget);
  const chargesMonomer = estimatePartialCharges(atomsMonomer);

  // Compute energy components
  const E_vdw = computeVdWEnergy(atomsTarget, atomsMonomer, cutoff_vdw);
  const E_elec = computeElectrostaticEnergy(atomsTarget, chargesTarget, atomsMonomer, chargesMonomer, cutoff_elec);

  const { energy: E_hbond, count: n_hbonds } = computeHBondEnergy(atomsTarget, atomsMonomer, cutoff_hbond);
  const { penalty: E_clash, count: n_clashes } = computeClashPenalty(atomsTarget, atomsMonomer, clash_threshold);

  const E_total = E_vdw + E_elec + E_hbond + E_clash;

  // Size normalization (like diffmip-lite)
  const n_target = atomsTarget.length;
  const n_monomer = atomsMonomer.length;
  const combined_size = Math.sqrt(n_target * n_monomer);
  const size_corrected_energy = combined_size > 0 ? E_total / combined_size : 0.0;

  return {
    energy_total: E_total,
    energy_vdw: E_vdw,
    energy_elec: E_elec,
    energy_hbond: E_hbond,
    energy_clash: E_clash,
    n_hbonds: n_hbonds,
    n_clashes: n_clashes,
    n_atoms_target: n_target,
    n_atoms_monomer: n_monomer,
    size_corrected_energy: size_corrected_energy,
    // Ligand efficiency (energy per heavy atom)
    ligand_efficiency: n_monomer > 0 ? E_total / n_monomer : 0.0,
  };
}

/**
 * Score a single monomer in the context of the complete system
 * Includes both target-monomer and monomer-monomer interactions
 */
function scoreMonomerInSystem(targetPDB, monomerStructure, otherMonomerStructures, options = {}) {
  const {
    cutoff_vdw = 8.0,
    cutoff_elec = 12.0,
    cutoff_hbond = 3.5,
    clash_threshold = 2.0
  } = options;

  // Parse atoms
  const atomsTarget = parseAtomsFromPDB(targetPDB);
  const atomsMonomer = parseAtomsFromSDF(monomerStructure) || parseAtomsFromPDB(monomerStructure);

  if (atomsTarget.length === 0 || atomsMonomer.length === 0) {
    return null;
  }

  // Estimate partial charges
  const chargesTarget = estimatePartialCharges(atomsTarget);
  const chargesMonomer = estimatePartialCharges(atomsMonomer);

  // 1. Compute target-monomer interactions
  let E_vdw_target = computeVdWEnergy(atomsTarget, atomsMonomer, cutoff_vdw);
  let E_elec_target = computeElectrostaticEnergy(atomsTarget, chargesTarget, atomsMonomer, chargesMonomer, cutoff_elec);
  let { energy: E_hbond_target, count: n_hbonds_target } = computeHBondEnergy(atomsTarget, atomsMonomer, cutoff_hbond);
  let { penalty: E_clash_target, count: n_clashes_target } = computeClashPenalty(atomsTarget, atomsMonomer, clash_threshold);

  // 2. Compute monomer-monomer interactions
  let E_vdw_mono = 0.0;
  let E_elec_mono = 0.0;
  let E_hbond_mono = 0.0;
  let n_hbonds_mono = 0;
  let E_clash_mono = 0.0;
  let n_clashes_mono = 0;

  for (const otherStructure of otherMonomerStructures) {
    if (!otherStructure) continue;

    const atomsOther = parseAtomsFromSDF(otherStructure) || parseAtomsFromPDB(otherStructure);
    if (atomsOther.length === 0) continue;

    const chargesOther = estimatePartialCharges(atomsOther);

    // Add pairwise interactions
    E_vdw_mono += computeVdWEnergy(atomsMonomer, atomsOther, cutoff_vdw);
    E_elec_mono += computeElectrostaticEnergy(atomsMonomer, chargesMonomer, atomsOther, chargesOther, cutoff_elec);

    const hbond_result = computeHBondEnergy(atomsMonomer, atomsOther, cutoff_hbond);
    E_hbond_mono += hbond_result.energy;
    n_hbonds_mono += hbond_result.count;

    const clash_result = computeClashPenalty(atomsMonomer, atomsOther, clash_threshold);
    E_clash_mono += clash_result.penalty;
    n_clashes_mono += clash_result.count;
  }

  // Total energies
  const E_vdw = E_vdw_target + E_vdw_mono;
  const E_elec = E_elec_target + E_elec_mono;
  const E_hbond = E_hbond_target + E_hbond_mono;
  const E_clash = E_clash_target + E_clash_mono;
  const n_hbonds = n_hbonds_target + n_hbonds_mono;
  const n_clashes = n_clashes_target + n_clashes_mono;

  const E_total = E_vdw + E_elec + E_hbond + E_clash;

  // Size normalization
  const n_target = atomsTarget.length;
  const n_monomer = atomsMonomer.length;
  const combined_size = Math.sqrt(n_target * n_monomer);
  const size_corrected_energy = combined_size > 0 ? E_total / combined_size : 0.0;

  return {
    energy_total: E_total,
    energy_vdw: E_vdw,
    energy_elec: E_elec,
    energy_hbond: E_hbond,
    energy_clash: E_clash,
    // Breakdown by interaction type
    energy_target_monomer: E_vdw_target + E_elec_target + E_hbond_target + E_clash_target,
    energy_monomer_monomer: E_vdw_mono + E_elec_mono + E_hbond_mono + E_clash_mono,
    n_hbonds: n_hbonds,
    n_clashes: n_clashes,
    n_clashes_with_target: n_clashes_target,
    n_clashes_with_monomers: n_clashes_mono,
    n_atoms_target: n_target,
    n_atoms_monomer: n_monomer,
    size_corrected_energy: size_corrected_energy,
    ligand_efficiency: n_monomer > 0 ? E_total / n_monomer : 0.0,
  };
}

// Export for use in diffmip-client.js
if (typeof window !== 'undefined') {
  window.PhysicsScoring = {
    scoreInteraction,
    scoreMonomerInSystem,
    parseAtomsFromPDB,
    parseAtomsFromSDF
  };
}

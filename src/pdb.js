/* ── openMIP PDB utilities ── */

// ── remapPDB ─────────────────────────────────────────────────────────────────
// Ports the logic from scratch/make_pdb_cartoon.py.
//
// Problem: 3Dmol.js cartoon/ribbon rendering requires backbone atoms
// (N, CA, C, O) to appear in canonical order within each residue record.
// Some PDB files have atoms out of order or with non-standard ordering,
// causing the ribbon trace to break silently.
//
// Fix: parse every ATOM/HETATM record, group by residue, sort each group
// so that atoms appear in the canonical N→CA→C→O→sidechain sequence, then
// reassemble with renumbered serials.  CONECT / REMARK / header lines are
// preserved verbatim (CONECT indices are not updated — strip them instead
// since 3Dmol recomputes bonds from proximity for standard residues).
function remapPDB(pdbText) {
  // Canonical heavy-atom order for each standard amino acid.
  // Matches the cartoon_res dict in make_pdb_cartoon.py.
  const CANONICAL = {
    ALA: ['N','CA','C','O','CB'],
    ARG: ['N','CA','C','O','CB','CG','CD','NE','CZ','NH1','NH2'],
    ASN: ['N','CA','C','O','CB','CG','OD1','ND2'],
    ASP: ['N','CA','C','O','CB','CG','OD1','OD2'],
    CYS: ['N','CA','C','O','CB','SG'],
    GLN: ['N','CA','C','O','CB','CG','CD','OE1','NE2'],
    GLU: ['N','CA','C','O','CB','CG','CD','OE1','OE2'],
    GLY: ['N','CA','C','O'],
    HIS: ['N','CA','C','O','CB','CG','ND1','CD2','CE1','NE2'],
    ILE: ['N','CA','C','O','CB','CG1','CG2','CD1'],
    LEU: ['N','CA','C','O','CB','CG','CD1','CD2'],
    LYS: ['N','CA','C','O','CB','CG','CD','CE','NZ'],
    MET: ['N','CA','C','O','CB','CG','SD','CE'],
    PHE: ['N','CA','C','O','CB','CG','CD1','CD2','CE1','CE2','CZ'],
    PRO: ['N','CA','C','O','CB','CG','CD'],
    SER: ['N','CA','C','O','CB','OG'],
    THR: ['N','CA','C','O','CB','OG1','CG2'],
    TRP: ['N','CA','C','O','CB','CG','CD1','CD2','NE1','CE2','CE3','CZ2','CZ3','CH2'],
    TYR: ['N','CA','C','O','CB','CG','CD1','CD2','CE1','CE2','CZ','OH'],
    VAL: ['N','CA','C','O','CB','CG1','CG2'],
  };

  const residueAtoms = new Map();   // key → [line, ...]
  const residueOrder = [];          // insertion-order list of keys
  const headerLines  = [];
  // CONECT lines are intentionally dropped: 3Dmol recomputes bonds from
  // proximity for standard residues, and stale serial numbers after
  // renumbering would corrupt them.

  for (const line of pdbText.split('\n')) {
    const rec = line.substring(0, 6).trim();
    if (rec === 'ATOM' || rec === 'HETATM') {
      // Residue key: chain (col 22) + sequence number (cols 23-26) + iCode (col 27)
      const key = line[21] + line.substring(22, 27);
      if (!residueAtoms.has(key)) {
        residueAtoms.set(key, []);
        residueOrder.push(key);
      }
      residueAtoms.get(key).push(line);
    } else if (rec !== 'CONECT' && rec !== 'END' && line.trim()) {
      headerLines.push(line);
    }
  }

  const out = [...headerLines];
  let serial = 1;

  for (const key of residueOrder) {
    const atoms   = residueAtoms.get(key);
    const resName = atoms[0].substring(17, 20).trim();
    const canon   = CANONICAL[resName];

    let sorted;
    if (canon) {
      // Build a rank map for O(1) lookup
      const rank = {};
      canon.forEach((name, i) => { rank[name] = i; });
      sorted = [...atoms].sort((a, b) => {
        const na = a.substring(12, 16).trim();
        const nb = b.substring(12, 16).trim();
        return (rank[na] ?? 9999) - (rank[nb] ?? 9999);
      });
    } else {
      // Unknown residue — keep original order
      sorted = atoms;
    }

    for (const line of sorted) {
      // Rewrite the serial number (cols 7-11) and keep everything else
      const newLine = line.substring(0, 6) +
                      String(serial).padStart(5) +
                      line.substring(11);
      out.push(newLine);
      serial++;
    }
  }

  out.push('END');
  return out.join('\n');
}

// ── computePDBStats ───────────────────────────────────────────────────────────
// Parse a PDB text and return { mw, atoms, heavy, formula }.
//
// MW strategy (per residue):
//   - Standard amino acid  → use AA residue mass; add 18.02 Da (terminal water) once.
//   - Everything else      → sum individual atomic masses.
//
// Formula:
//   - Peptide/protein (≥1 AA residue) → "<N> residues · chain(s) X"
//   - Small molecule                  → Hill-order empirical formula (C, H, then α)

// Average residue masses (Da) for standard amino acids.
// Each value is the residue weight (full AA mass minus water).
const AA_RESIDUE_MASS = {
  ALA:  71.08,  ARG: 156.19, ASN: 114.10, ASP: 115.09, CYS: 103.14,
  GLN: 128.13,  GLU: 129.12, GLY:  57.05, HIS: 137.14, ILE: 113.16,
  LEU: 113.16,  LYS: 128.17, MET: 131.20, PHE: 147.18, PRO:  97.12,
  SER:  87.08,  THR: 101.10, TRP: 186.21, TYR: 163.18, VAL:  99.13,
};

// Average atomic masses (Da) for elements found in biomolecules.
const ATOM_MASS = {
  H: 1.008,   C: 12.011,  N: 14.007,  O: 15.999,  S: 32.06,
  P: 30.974,  F: 18.998,  CL: 35.45,  BR: 79.904,  I: 126.904,
  FE: 55.845, CA: 40.078, MG: 24.305, ZN: 65.38,  NA: 22.990,
  K:  39.098, SE: 78.97,
};

function computePDBStats(pdbText) {
  const residues     = new Map();   // residueKey → { resname, elements: string[] }
  const residueOrder = [];

  for (const line of pdbText.split('\n')) {
    const rec = line.substring(0, 6).trim();
    if (rec !== 'ATOM' && rec !== 'HETATM') continue;

    const resName    = line.substring(17, 20).trim();
    const chain      = line[21] || ' ';
    const resSeq     = line.substring(22, 27).trim();
    const atomName   = line.substring(12, 16).trim();
    // Prefer the dedicated element column (cols 77-78); fall back to atom name.
    const elemCol    = line.length > 76 ? line.substring(76, 78).trim() : '';
    const element    = (elemCol || atomName.replace(/[^A-Za-z]/g, '')[0] || 'C').toUpperCase();

    const key = chain + '\x00' + resSeq;
    if (!residues.has(key)) {
      residues.set(key, { resname: resName, elements: [] });
      residueOrder.push(key);
    }
    residues.get(key).elements.push(element);
  }

  const residueList = residueOrder.map(k => residues.get(k));

  // Atom counts
  let totalAtoms = 0;
  let heavyAtoms = 0;
  const elemCount = {};
  residueList.forEach(r => r.elements.forEach(el => {
    totalAtoms++;
    if (el !== 'H' && el !== 'D') heavyAtoms++;
    elemCount[el] = (elemCount[el] || 0) + 1;
  }));

  // MW — hybrid per-residue strategy
  let mw = 0;
  let aaCount = 0;
  residueList.forEach(r => {
    if (r.resname in AA_RESIDUE_MASS) {
      mw += AA_RESIDUE_MASS[r.resname];
      aaCount++;
    } else {
      r.elements.forEach(el => { mw += ATOM_MASS[el] || 0; });
    }
  });
  if (aaCount > 0) mw += 18.02;   // peptide N/C terminus

  // Formula
  const toSub = n => n < 2 ? '' :
    n.toString().split('').map(d => '₀₁₂₃₄₅₆₇₈₉'[+d]).join('');

  let formula;
  if (aaCount > 0) {
    const chains = [...new Set(
      residueOrder.map(k => k.split('\x00')[0]).filter(c => c.trim())
    )];
    formula = `${aaCount} residue${aaCount !== 1 ? 's' : ''}`;
    if (chains.length) {
      formula += ` · chain${chains.length > 1 ? 's' : ''} ${chains.join(', ')}`;
    }
  } else {
    // Hill notation: C, H first, then remaining elements alphabetically
    const HILL_FIRST = ['C', 'H', 'N', 'O', 'P', 'S'];
    const allElem = [
      ...HILL_FIRST.filter(e => elemCount[e]),
      ...Object.keys(elemCount).filter(e => !HILL_FIRST.includes(e)).sort(),
    ];
    formula = allElem.map(e => e + toSub(elemCount[e])).join('');
  }

  return {
    mw:    mw > 0 ? mw.toFixed(2) + ' g/mol' : '—',
    atoms: String(totalAtoms),
    heavy: String(heavyAtoms),
    formula,
  };
}

#!/usr/bin/env python3

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.align import alignto
from MDAnalysis.coordinates.GRO import GROWriter
from pathlib import Path
import os
import shutil
import subprocess


# =========================
# Configuration
# =========================
ketMD_dir = Path('/Users/jlmr/Downloads/ketMD_test')
input_files = ketMD_dir / 'input_files'
first_cycle = 0

start_file = input_files / 'step6.7_equilibration.gro'
target_file = input_files / 'ABCG2-OF.gro'

topol_file = input_files / 'topol.top'
mdp_file = input_files / 'mdp.mdp'
ndx_file = input_files / 'index.ndx'

protein_sele = "protein"
ligand_sele = "resname LIG"

velocity_increase_factor = 1.08
output_file = ketMD_dir / input_files / "perturbed_start.gro"
# =========================


def prepare_cycle(k):
    # first cycle
    if k == 0:
        cycle_dir = ketMD_dir / f'cycle{k}'
        os.makedirs(cycle_dir)
        shutil.copy(output_file, cycle_dir / 'before_md.gro')
    
    # run grompp 
    cmd = ['gmx', 'grompp', 
           '-f', mdp_file, '-p', topol_file,
           '-c', cycle_dir / 'before_md.gro',
           '-n', ndx_file, '-o', cycle_dir / 'tpr.tpr']
    subprocess.run(cmd)


prepare_cycle(k=0)



def main():

    u_target = mda.Universe(target_file)
    u_md = mda.Universe(start_file)

    if u_md.trajectory.ts.velocities is None:
        raise ValueError("Start GRO file does not contain velocities.")

    protein_target = u_target.select_atoms(protein_sele)
    protein_md = u_md.select_atoms(protein_sele)
    ligand_md = u_md.select_atoms(ligand_sele)

    print(f"[INFO] Protein atoms: {len(protein_md)}")
    print(f"[INFO] Ligand atoms:  {len(ligand_md)}")

    # =========================
    # Align
    # =========================
    alignto(protein_md, protein_target)

    # =========================
    # Grab velocity array
    # =========================
    velocities = u_md.trajectory.ts.velocities.copy()

    protein_indices = protein_md.indices
    ligand_indices = ligand_md.indices

    # -------------------------
    # DEBUG BEFORE
    # -------------------------
    print("\nDEBUG BEFORE:")
    print("First protein velocity:",
          velocities[protein_indices[0]])
    print("First ligand velocity:",
          velocities[ligand_indices[0]])

    # =========================
    # 1️⃣ Protein perturbation
    # =========================
    pos_md = protein_md.positions
    pos_target = protein_target.positions

    delta = pos_target - pos_md
    delta_unit = delta / np.linalg.norm(delta)

    vel_protein = velocities[protein_indices]

    projection = np.sum(vel_protein * delta_unit)

    vel_protein_new = vel_protein + \
        (velocity_increase_factor - 1.0) * projection * delta_unit

    velocities[protein_indices] = vel_protein_new

    # =========================
    # 2️⃣ Ligand z-perturbation
    # =========================
    velocities[ligand_indices, 2] *= velocity_increase_factor

    # -------------------------
    # DEBUG AFTER
    # -------------------------
    print("\nDEBUG AFTER:")
    print("First protein velocity:",
          velocities[protein_indices[0]])
    print("First ligand velocity:",
          velocities[ligand_indices[0]])

    max_change = np.max(np.abs(
        velocities - u_md.trajectory.ts.velocities
    ))

    print("\nMax absolute velocity change in system:", max_change)

    # =========================
    # Assign back + write
    # =========================
    u_md.trajectory.ts.velocities = velocities

    with GROWriter(output_file, reindex=False) as writer:
        writer.write(u_md.atoms)

    print(f"\n[SUCCESS] Written to:\n{output_file}")


#main()
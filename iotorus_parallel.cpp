//========================================================================================
// Io Torus Problem Generator
// PHYSICAL MECHANICS: Phipps & Withers (2017) Empirical Data Assimilation
//========================================================================================

#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../parameter_input.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../stable_profile.h"
#include <cmath>
#include <algorithm> // Required for std::max safety floor
#include <iostream>
#include <fstream>
#include <iomanip>

Real Omega_J;
Real breakdown_radius;
Real start_io_phase_rad;
Real start_cml_rad;

// --- TRACER CONTROLS ---
bool enable_tracer;
Real tracer_start;
Real tracer_duration;

// --- NEW GLOBALS FOR THERMODYNAMICS ---
bool enable_temp_nudge;
Real global_B0;

// --- NEW GLOBALS FOR PHIPPS PROFILE & DYNAMICS ---
Real N1_val, C1_val, W1_val;
Real N2_val, C2_val, W2_val;
Real N3_val, C3_val, W3_val;
Real N4_val, C4_val, W4_val;
Real cliff_radius, relax_cliff, relax_torus;
Real recomb_rate;

void MeshBlock::UserWorkInLoop(void) {
    Real max_wake_rho = 0.0;
    Real wake_vphi = 0.0;
    Real io_radius = 5.9;

    // --- NEW: Plasma Probe Variables ---
    Real probe_rho = 0.0;
    Real probe_Bz = 0.0;
    Real probe_vA = 0.0;
    bool probe_triggered = false;

// =======================================================================
    // THE RIGID DIPOLE LOCK (Prevents Magnetic Washout)
    // =======================================================================
    if (MAGNETIC_FIELDS_ENABLED) {
        for (int k=ks; k<=ke; ++k) {
            for (int j=js; j<=je; ++j) {
                for (int i=is; i<=ie; ++i) {
                    Real r = pcoord->x1v(i);

                    // 1. Force the cell-centered fields back to the absolute dipole
                    pfield->bcc(IB1,k,j,i) = 0.0;
                    pfield->bcc(IB2,k,j,i) = 0.0;
                    pfield->bcc(IB3,k,j,i) = global_B0 / (r*r*r); // <-- UPDATED

                    // 2. Force the face-centered fields (to maintain divergence-free CT)
                    pfield->b.x1f(k,j,i) = 0.0;
                    pfield->b.x2f(k,j,i) = 0.0;
                    pfield->b.x3f(k,j,i) = global_B0 / (r*r*r); // <-- UPDATED

                    // 3. Patch the outer boundary faces of the cells
                    if (i == ie) pfield->b.x1f(k,j,i+1) = 0.0;
                    if (j == je) pfield->b.x2f(k,j+1,i) = 0.0;
                    if (k == ke) pfield->b.x3f(k+1,j,i) = global_B0 / (r*r*r); // <-- UPDATED
                }
            }
        }
    }

    // =======================================================================
    // DIAGNOSTIC SCANNER
    // =======================================================================
    for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
            for (int i=is; i<=ie; ++i) {
                Real r = pcoord->x1v(i);

                // A. Track the Io Wake
                if (std::abs(r - io_radius) < 0.2) {
                    Real current_rho = phydro->u(IDN,k,j,i);
                    if (current_rho > max_wake_rho) {
                        max_wake_rho = current_rho;
                        wake_vphi = phydro->u(IM2,k,j,i) / current_rho;
                    }
                }

                // B. LIVE PLASMA PROBE (Drop a sensor at ~6.0 R_J)
                if (!probe_triggered && std::abs(r - 6.0) < 0.1) {
                    probe_rho = phydro->u(IDN,k,j,i);
                    if (MAGNETIC_FIELDS_ENABLED) {
                        probe_Bz = pfield->bcc(IB3,k,j,i);
                        probe_vA = std::sqrt( (probe_Bz * probe_Bz) / (probe_rho + 1.0e-10) );
                    }
                    probe_triggered = true;
                }
            }
        }
    }

    // Calculate Radio Proxy & Write to CSV
    Real target_corotation = io_radius * Omega_J;
    Real velocity_shear = target_corotation - wake_vphi;
    Real alfven_power_proxy = max_wake_rho * SQR(velocity_shear);

    static bool is_first_run = true;
    std::ofstream out_file("wake_radio_diagnostics.csv", std::ios_base::app);

    if (out_file.is_open()) {
        if (is_first_run) {
            out_file << "Time,Max_Wake_Density,Velocity_Shear,Alfven_Power_Proxy,Probe_Rho,Probe_Bz,Probe_vA\n";
            is_first_run = false;
        }
        out_file << std::fixed << std::setprecision(5)
                 << pmy_mesh->time << ","
                 << max_wake_rho << ","
                 << velocity_shear << ","
                 << alfven_power_proxy << ","
                 << probe_rho << ","
                 << probe_Bz << ","
                 << probe_vA << "\n";
        out_file.close();
    }
}

// --- Function Prototypes ---
Real GetPhippsDensity(Real r);
void IoSourceTerm(MeshBlock *pmb, const Real time, const Real dt,
                  const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
                  const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
                  AthenaArray<Real> &cons_scalar);
void RunDiagnostics(MeshBlock *pmb, Real time, Real dt, AthenaArray<Real> &cons);

//========================================================================================
void Mesh::InitUserMeshData(ParameterInput *pin) {
  EnrollUserExplicitSourceFunction(IoSourceTerm);
  Omega_J = pin->GetOrAddReal("problem", "omega_j", 1.0);
  breakdown_radius = pin->GetOrAddReal("problem", "breakdown_radius", 20.0);

  // Load Phipps & Withers (2017) Profile Parameters
  N1_val = pin->GetOrAddReal("problem", "N1", 1710.0);
  C1_val = pin->GetOrAddReal("problem", "C1", 5.23);
  W1_val = pin->GetOrAddReal("problem", "W1", 0.20);

  N2_val = pin->GetOrAddReal("problem", "N2", 2180.0);
  C2_val = pin->GetOrAddReal("problem", "C2", 5.60);
  W2_val = pin->GetOrAddReal("problem", "W2", 0.08);

  N3_val = pin->GetOrAddReal("problem", "N3", 2160.0);
  C3_val = pin->GetOrAddReal("problem", "C3", 5.89);
  W3_val = pin->GetOrAddReal("problem", "W3", 0.32);

  N4_val = pin->GetOrAddReal("problem", "N4", 1601.0);
  C4_val = pin->GetOrAddReal("problem", "C4", 5.52);
  W4_val = pin->GetOrAddReal("problem", "W4", 1.88);

  // Load Dynamics & Spring Parameters
  cliff_radius = pin->GetOrAddReal("problem", "cliff_radius", 4.8);
  relax_cliff  = pin->GetOrAddReal("problem", "relax_cliff", 0.1);
  relax_torus  = pin->GetOrAddReal("problem", "relax_torus", 3.0);
  recomb_rate  = pin->GetOrAddReal("problem", "recomb_rate", 0.1);

  // Load Tracer Parameters
  enable_tracer   = pin->GetOrAddBoolean("problem", "enable_tracer", false);
  tracer_start    = pin->GetOrAddReal("problem", "tracer_start", 5.0);
  tracer_duration = pin->GetOrAddReal("problem", "tracer_duration", 1.0);

  // Load Thermo Toggle (Defaults to true/ON if not specified)
  enable_temp_nudge = pin->GetOrAddBoolean("problem", "enable_temp_nudge", true);

  // Load Global Magnetic Field
  global_B0 = pin->GetOrAddReal("problem", "B0", 5000.0);

  //Io and CML Phase Parameters
  start_io_phase_rad = pin->GetOrAddReal("problem", "start_io_phase", 90.0) * (M_PI / 180.0);
  start_cml_rad      = pin->GetOrAddReal("problem", "start_cml", 120.0) * (M_PI / 180.0);

}
//========================================================================================
// Hiraki et al. (2012) Analytical Radial Confinement (Eq. 5)
//========================================================================================
Real GetHirakiDensity(Real r) {
    Real r_io = 6.0; // The paper centers the torus exactly at 6.0 R_J

    // The paper restricts plasmas to a region symmetric around Io's orbit
    // with a width of 10 R_Io (where 1 R_Io = a/39, making delta approx 0.256 R_J)
    Real delta = 10.0 / 39.0;

    if (std::abs(r - r_io) <= delta) {
        Real cos_val = std::cos((M_PI * (r - r_io)) / (2.0 * delta));
        // Scales to the peak density of 2000 cm^-3 used in the paper
        return 2000.0 * (cos_val * cos_val);
    }
    return 1.0e-3; // Background vacuum floor
}

//========================================================================================
// Hiraki et al. (2012) Interchange Instability Seed (Eq. 6)
//========================================================================================
Real GetHirakiPerturbation(Real phi) {
    Real sum = 0.0;

    // The paper assumes a sum of interchange unstable toroidal modes from m=1 to 50
    for (int m = 1; m <= 50; ++m) {
        // We use a deterministic golden-angle phase shift for gamma(m)
        // to ensure grid stability across parallel MPI boundaries.
        Real gamma = (Real)m * 2.39996;
        sum += (1.0 / (Real)m) * std::sin((Real)m * phi - gamma);
    }

    // The paper uses an amplitude of 10^-2 (1%) for the perturbation
    return 1.0 + (0.01 * sum);
}

//========================================================================================
// The Phipps and Withers (2017) Empirical Profile
//========================================================================================
Real GetPhippsDensity(Real r) {
    Real n_val = 0.0;

    // THE CENTRIFUGAL CLIFF
    if (r < cliff_radius) {
        n_val = 3.0; // Sets the flat, empty floor
    }
    // The Main Torus Peaks
    else if (r < 6.1) {
        n_val = N1_val * std::exp(-SQR((r - C1_val) / W1_val)) +
                N2_val * std::exp(-SQR((r - C2_val) / W2_val)) +
                N3_val * std::exp(-SQR((r - C3_val) / W3_val));
    }
    // The Extended Tail
    else {
        n_val = N4_val * std::exp(-SQR((r - C4_val) / W4_val));
    }

    Real scale_factor = 10000.0 / 3000.0;
    Real final_density;
    if (r < cliff_radius) {
        final_density = n_val;
    } else {
        final_density = n_val * scale_factor;
    }

    if (final_density < 1.0e-3) final_density = 1.0e-3;
    return final_density;
}

//========================================================================================
// The Voyager 1 Empirical Temperature Profile (in eV)
//========================================================================================
Real GetVoyagerTemperature(Real r) {
    if (r < 5.8) return 5.0; // Cold inner core
    if (r < 6.0) return 5.0 + (60.0 - 5.0) * ((r - 5.8) / 0.2); // Steep jump at Io
    // Gradual climb to 300 eV at 10 R_J
    Real temp = 60.0 + (300.0 - 60.0) * ((r - 6.0) / 4.0);
    return std::min(temp, 500.0); // Safety cap
}

//========================================================================================
// Problem Generator: Warm Start (Loaded from stable_profile.h)
//========================================================================================
void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  // --- THE SWITCH ---
  // Default to 'false' (0) if the parameter is missing from athinput
  bool use_stable = pin->GetOrAddBoolean("problem", "use_stable_profile", false);

  #if NON_BAROTROPIC_EOS
    Real gamma = peos->GetGamma();
    Real gm1 = gamma - 1.0;
  #endif

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        Real r = pcoord->x1v(i);
        Real local_rho, local_press, local_vphi;

        if (use_stable) {
            // --- WARM START LOGIC (Your existing interpolation) ---
            int idx = 0;
            while (idx < STABLE_SIZE - 2 && stable_r[idx + 1] < r) idx++;
            Real frac = (r - stable_r[idx]) / (stable_r[idx + 1] - stable_r[idx]);

            local_rho   = stable_rho[idx] + frac * (stable_rho[idx+1] - stable_rho[idx]);
            local_press = stable_press[idx] + frac * (stable_press[idx+1] - stable_press[idx]);
            local_vphi  = stable_vphi[idx] + frac * (stable_vphi[idx+1] - stable_vphi[idx]);
        } else {
            // --- COLD START LOGIC ---
            local_rho   = 0.1;
            local_press = 1.0e-5;

            // Calculate the physical Hill profile at this exact radius
            Real hill_ratio = 1.0 / (1.0 + SQR(r / breakdown_radius));
            local_vphi  = r * (Omega_J * hill_ratio);
        }

        // Apply shared safety floors and physics assignments
        local_rho   = std::max(local_rho, 1.0e-5);
        local_press = std::max(local_press, 1.0e-6);

        phydro->u(IDN,k,j,i) = local_rho;
        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = local_rho * local_vphi;
        phydro->u(IM3,k,j,i) = 0.0;

        Real e_mag = 0.0;
        if (MAGNETIC_FIELDS_ENABLED) {
            // 1. Set the Face-Centered Magnetic Fields
            pfield->b.x1f(k,j,i) = 0.0;
            pfield->b.x2f(k,j,i) = 0.0;
            pfield->b.x3f(k,j,i) = global_B0 / (r*r*r); // <-- UPDATED

            // Set boundary faces
            if (i == ie) pfield->b.x1f(k,j,i+1) = 0.0;
            if (j == je) pfield->b.x2f(k,j+1,i) = 0.0;
            if (k == ke) pfield->b.x3f(k+1,j,i) = global_B0 / (r*r*r); // <-- UPDATED

            // 2. Set the Cell-Centered averages
            pfield->bcc(IB1,k,j,i) = 0.0;
            pfield->bcc(IB2,k,j,i) = 0.0;
            pfield->bcc(IB3,k,j,i) = global_B0 / (r*r*r); // <-- UPDATED

            e_mag = 0.5 * SQR(pfield->bcc(IB3,k,j,i));
        }

        #if NON_BAROTROPIC_EOS
           Real e_kin = 0.5 * local_rho * SQR(local_vphi);
           phydro->u(IEN,k,j,i) = (local_press / gm1) + e_kin + e_mag;
        #endif
      }
    }
  }
}

//========================================================================================
// Source Term: Data Assimilation + DYNAMIC IO WAKE
//========================================================================================
void IoSourceTerm(MeshBlock *pmb, const Real time, const Real dt,
                  const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
                  const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
                  AthenaArray<Real> &cons_scalar) {

  // Calculate Io's exact position with customizable starting phase
  Real Omega_Io = Omega_J * 0.2336;
  Real phi_io = std::fmod(start_io_phase_rad + Omega_Io * time, 2.0 * M_PI);
  if (phi_io < 0.0) phi_io += 2.0 * M_PI;

  const Real RHO_FLOOR = 1.0e-3;
  Real io_radius = 5.9;
  Real io_width = 0.5;

  // Calculate customizable System III CML phase
  Real sys3_phase = std::fmod(start_cml_rad + Omega_J * time, 2.0 * M_PI);

  // --- 3D TILT PROXY: Synodic Modulation ---
  Real Omega_rel = Omega_J - Omega_Io;
  Real phase_offset = 0.0;
  Real tilt_wave = std::pow(std::cos(Omega_rel * time + phase_offset), 2);

  // --- FIX 1: CALIBRATED MASS INJECTION ---
  // Lowered from 25000 to prevent the massive 15,000 cm^-3 density spikes
  Real mdot_max = 5000.0; // Calibrated for ~4000 cm^-3 total wake density
  Real mdot_min = 1000.0; // Accurate 3D tilt drop-off

  Real mdot_io = mdot_min + (mdot_max - mdot_min) * tilt_wave;

  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      for (int i=pmb->is; i<=pmb->ie; ++i) {

        Real r   = pmb->pcoord->x1v(i);
        Real phi = pmb->pcoord->x2v(j);
        Real rho = prim(IDN,k,j,i);

        // =======================================================================
        // 1. DYNAMIC IO INJECTION
        // =======================================================================
        Real dphi = std::abs(phi - phi_io);
        if (dphi > M_PI) dphi = 2.0 * M_PI - dphi;
        Real arc_len = r * dphi;
        Real dr      = r - io_radius;
        Real dist_sq = SQR(dr) + SQR(arc_len);

        if (dist_sq < 0.25) {
            Real source_rate = mdot_io * std::exp(-dist_sq / SQR(io_width));
            Real drho_io = source_rate * dt;
            Real v_inj_io = r * Omega_J;

            cons(IDN,k,j,i) += drho_io;
            cons(IM2,k,j,i) += drho_io * v_inj_io;

            if (enable_tracer && time >= tracer_start && time <= (tracer_start + tracer_duration)) {
                 cons_scalar(0,k,j,i) += drho_io;
            }

            #if NON_BAROTROPIC_EOS
                Real gamma = pmb->peos->GetGamma();
                Real v_pickup_io = (r * Omega_J) - v_inj_io;
                Real specific_heat_io = 0.5 * SQR(v_pickup_io);
                cons(IEN,k,j,i) += ((drho_io * specific_heat_io) / (gamma - 1.0)) + (0.5 * drho_io * SQR(v_inj_io));
            #endif

            rho += drho_io;
        }

        // =======================================================================
        // 2. TRUE DYNAMIC BACKGROUND
        // =======================================================================
        Real target_rho = GetPhippsDensity(r);
        Real relax_time = (r < cliff_radius) ? relax_cliff : relax_torus;

        Real noise_seed = GetHirakiPerturbation(phi);
        target_rho *= noise_seed;

        Real drho_force = 0.0;
        Real drho_loss  = 0.0;

        if (rho < target_rho) {
            drho_force = (target_rho - rho) * (dt / relax_time);
        } else {
            Real excess_rho = rho - target_rho;
            drho_loss = -excess_rho * recomb_rate * dt;
        }

        Real total_drho = drho_force + drho_loss;
        Real rho_new = rho + total_drho;
        rho_new = std::max(rho_new, 1.0e-5);

        Real hill_ratio = 1.0 / (1.0 + SQR(r / breakdown_radius));
        Real target_vphi = r * (Omega_J * hill_ratio);
        Real current_vphi = prim(IM2,k,j,i);

        Real v_mixed = (total_drho > 0.0) ?
                       (rho * current_vphi + total_drho * target_vphi) / rho_new :
                       current_vphi;
        Real v_final = v_mixed + (target_vphi - v_mixed) * (dt / relax_time);

        // E. Apply Final Hydrodynamic State
        cons(IDN,k,j,i) = rho_new;
        cons(IM2,k,j,i) = rho_new * v_final;

        // --- FIX 2: THE 2.5D MAGNETIC TENSION PROXY ---
        // Cancels the artificial B_z explosion so the plasma stays in orbit
        if (MAGNETIC_FIELDS_ENABLED) {
            Real artificial_outward_force = 3.0 * (global_B0 * global_B0) / std::pow(r, 7);

            // Subtract the artificial outward momentum
            cons(IM1,k,j,i) -= artificial_outward_force * dt;

            #if NON_BAROTROPIC_EOS
                // Remove the artificial work done from the energy state
                Real v_r = prim(IVX,k,j,i);
                cons(IEN,k,j,i) -= (artificial_outward_force * v_r) * dt;
            #endif
        }

        // F. Instant Energy Assimilation (The Voyager Temperature Target)
        #if NON_BAROTROPIC_EOS
            Real gamma = pmb->peos->GetGamma();
            Real b_sq = 0.0;

            if (MAGNETIC_FIELDS_ENABLED) {
                b_sq = SQR(bcc(IB1,k,j,i)) + SQR(bcc(IB2,k,j,i)) + SQR(bcc(IB3,k,j,i));
            }

            if (enable_temp_nudge) {
                Real target_T_eV = GetVoyagerTemperature(r);
                Real ev_to_code = 1.0 / 32.7;
                Real target_u_int = (target_T_eV * ev_to_code) / (gamma - 1.0) * rho_new;
                Real new_ke = 0.5 * rho_new * SQR(v_final);
                cons(IEN,k,j,i) = target_u_int + new_ke + (0.5 * b_sq);
            }
        #endif

        // 3. THERMAL SAFETY FLOOR & ALFVÉN LIMITER
        Real B_local = global_B0 / (r*r*r);
        Real max_vA = 200.0;
        Real safe_rho = (B_local * B_local) / (max_vA * max_vA);
        Real effective_floor = (safe_rho > 0.1) ? safe_rho : 0.1;

        if (cons(IDN,k,j,i) <= effective_floor || std::isnan(cons(IDN,k,j,i))) {
             cons(IDN,k,j,i) = effective_floor;
        }

        #if NON_BAROTROPIC_EOS
             Real mom_sq = SQR(cons(IM1,k,j,i)) + SQR(cons(IM2,k,j,i)) + SQR(cons(IM3,k,j,i));
             Real e_k = 0.5 * mom_sq / cons(IDN,k,j,i);
             Real e_int_floor = 1.0e-5 / (gamma - 1.0);

             if (cons(IEN,k,j,i) - e_k - (0.5 * b_sq) < e_int_floor) {
                 cons(IEN,k,j,i) = e_int_floor + e_k + (0.5 * b_sq);
             }
         #endif
      }
    }
  }
}

//========================================================================================
void RunDiagnostics(MeshBlock *pmb, Real time, Real dt, AthenaArray<Real> &cons) { }

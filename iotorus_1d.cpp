//========================================================================================
// Io Torus Problem Generator with Toggles for Injection, Drag, Diffusion, Losses, Asymmetry
//========================================================================================

#include "../athena.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../hydro/hydro.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include <cmath>
#include <iostream>

// --- Global toggles and parameters ---
static bool enable_injection, enable_diffusion, enable_drag, enable_losses, enable_dawn_dusk, enable_gravity;
static Real s_io, src_sigma, src_amp_total;
static Real f_O, f_S;
static Real u0, u_corot0, tau_u;
static Real E_pickup;
static Real D_r;
static Real tau_loss_O, tau_loss_S, tau_loss_rho;
static Real A_dd, omega_rot;
static Real GM_custom, g_taper_width;
static Real Omega_Io;    // Io orbital angular speed
static Real sigma_phi;   // azimuthal width of Io source hotspot

static bool diag_mass, diag_species, diag_energy, diag_Bfield;

//----------------------------------------------------------------------------------------
// ProblemGenerator: initialize hydro + scalars
//----------------------------------------------------------------------------------------
void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  // Read toggles
  enable_injection = pin->GetOrAddInteger("problem","enable_injection",1);
  enable_diffusion = pin->GetOrAddInteger("problem","enable_diffusion",0);
  enable_drag      = pin->GetOrAddInteger("problem","enable_drag",0);
  enable_losses    = pin->GetOrAddInteger("problem","enable_losses",0);
  enable_dawn_dusk = pin->GetOrAddInteger("problem","enable_dawn_dusk",0);
  enable_gravity   = pin->GetOrAddInteger("problem","enable_gravity",0);

  // Read parameters
  s_io          = pin->GetOrAddReal("problem","s_io",5.9);
  src_sigma     = pin->GetOrAddReal("problem","src_sigma",0.4);
  src_amp_total = pin->GetOrAddReal("problem","src_amp_total",0.018);
  f_O           = pin->GetOrAddReal("problem","f_O",0.6);
  f_S           = pin->GetOrAddReal("problem","f_S",0.4);
  u0            = pin->GetOrAddReal("problem","u0",0.05);
  u_corot0      = pin->GetOrAddReal("problem","u_corot0",0.1);
  tau_u         = pin->GetOrAddReal("problem","tau_u",200.0);
  E_pickup      = pin->GetOrAddReal("problem","E_pickup",1.0);
  D_r           = pin->GetOrAddReal("problem","D_r",1e-3);
  tau_loss_O    = pin->GetOrAddReal("problem","tau_loss_O",300.0);
  tau_loss_S    = pin->GetOrAddReal("problem","tau_loss_S",600.0);
  tau_loss_rho  = pin->GetOrAddReal("problem","tau_loss_rho",5500.0);
  A_dd          = pin->GetOrAddReal("problem","A_dd",0.18);
  omega_rot     = pin->GetOrAddReal("problem","omega_rot",1.0);
  GM_custom     = pin->GetOrAddReal("problem","GM_custom",0.02);
  g_taper_width = pin->GetOrAddReal("problem","g_taper_width",0.8);
  Omega_Io      = pin->GetOrAddReal("problem","Omega_Io",0.4);   // code units
  sigma_phi     = pin->GetOrAddReal("problem","sigma_phi",0.35); // radians (~20°)

  Real rho0 = pin->GetOrAddReal("problem","rho0",1e-3);
  Real p0   = pin->GetOrAddReal("problem","p0",1.0);

  diag_mass    = pin->GetOrAddInteger("problem","diag_mass",1);
  diag_species = pin->GetOrAddInteger("problem","diag_species",1);
  diag_energy  = pin->GetOrAddInteger("problem","diag_energy",0);
  diag_Bfield  = pin->GetOrAddInteger("problem","diag_Bfield",0);

  // Passive scalar seeds
  const Real r0_seed_amp = 1e-12;   // oxygen seed amplitude
  const Real r1_seed_amp = 2e-12;   // sulfur seed amplitude
  const Real r0_center   = 5.9;     // near Io
  const Real r1_center   = 7.0;     // slightly outward
  const Real r_sigma     = 0.2;     // narrow Gaussian width

  // --- Background corotation with tunable sub-corotation lag ---
  auto vphi_corot = [&](Real r) {
    Real v_corot = omega_rot * r;
    Real v_lag   = u0 * (1.0 - std::exp(-r / tau_u)) * r;
    return v_corot - v_lag;
  };

  // Initialize hydro + scalars
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        Real r = pcoord->x1v(i);

        phydro->u(IDN,k,j,i) = rho0;
        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = 0.0;

        Real vphi0 = vphi_corot(r);
        phydro->u(IM3,k,j,i) = rho0 * vphi0;

#if NON_BAROTROPIC_EOS
        Real ek  = 0.5*(SQR(phydro->u(IM1,k,j,i)) + SQR(phydro->u(IM3,k,j,i)))/rho0;
        Real gm1 = peos->GetGamma() - 1.0;
        phydro->u(IEN,k,j,i) = p0/gm1 + ek;
#endif

        Real r0_seed = r0_seed_amp * std::exp(-SQR((r - r0_center)/r_sigma));
        Real r1_seed = r1_seed_amp * std::exp(-SQR((r - r1_center)/r_sigma));
        phydro->u(NHYDRO+0,k,j,i) = r0_seed;
        phydro->u(NHYDRO+1,k,j,i) = r1_seed;
      }
    }
  }

#ifdef MAGNETIC_FIELDS
  Field *pf = pfield;
  const Real B0 = pin->GetOrAddReal("problem","B0",2e-6); // Tesla units
  const Real theta_eq = M_PI/2.0;

  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        Real rf  = pcoord->x1f(i);
        Real thf = theta_eq;

        Real invr3   = 1.0/(rf*rf*rf);
        Real cost    = std::cos(thf);
        Real sint    = std::sin(thf);

        Real Br_face  = B0 * 2.0 * cost * invr3;
        Real Bth_face = B0 * sint * invr3;
        Real Bph_face = 0.0;

        pf->b.x1f(k,j,i) = Br_face;
        pf->b.x2f(k,j,i) = Bth_face;
        pf->b.x3f(k,j,i) = Bph_face;
      }
    }
  }
#endif
}



//========================================================================================
// UserSource: Io Plasma Torus sources/sinks with hotspot, pickup, drag, diffusion, losses
//========================================================================================

// --- Helper: wrap angle to [-pi, pi] ---
inline Real wrap_pi(Real dphi) {
  while (dphi >  M_PI) dphi -= 2.0*M_PI;
  while (dphi < -M_PI) dphi += 2.0*M_PI;
  return dphi;
}

void UserSource(MeshBlock *pmb, Real time, Real dt,
                const AthenaArray<Real> &prim,
                const AthenaArray<Real> &bcc,
                const AthenaArray<Real> &cons_in,
                AthenaArray<Real> &cons_out,
                AthenaArray<Real> &src) {
  auto &u = cons_out;
  Coordinates *coord = pmb->pcoord;
  EquationOfState *peos = pmb->peos;

  // Ramp up sources smoothly
  Real ramp = (time <= 0.0) ? 0.0 : std::min(1.0, time/150.0);

  // Floors
  const Real rho_floor = 1e-30;
  const Real r_floor   = 1e-30;

  // Io orbital phase
  Real phi_Io = std::fmod(Omega_Io * time, 2.0*M_PI);

  // --- Main source/sink application ---
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        Real r   = coord->x1v(i);
        Real phi = (pmb->block_size.nx3 > 1) ? coord->x3v(i) : 0.0; // azimuth if resolved

        // --- Injection localized at Io ---
        if (enable_injection) {
          // Radial shaping
          Real gO = std::exp(-SQR((r - s_io) / src_sigma));
          Real gS = std::exp(-SQR((r - (s_io + 0.2)) / (src_sigma * 1.25)));
          Real tailO = 0.10 * std::exp(-SQR((r - s_io) / 1.0));
          Real tailS = 0.20 * std::exp(-SQR((r - s_io) / 1.2));

          // Azimuthal hotspot
          Real dphi   = wrap_pi(phi - phi_Io);
          Real w_phi  = std::exp(-SQR(dphi) / (2.0 * SQR(sigma_phi)));

          // Dawn–dusk asymmetry tied to longitude (optional)
          Real dd = enable_dawn_dusk ? (1.0 + A_dd * std::sin(wrap_pi(phi))) : 1.0;

          // Species-specific sources
          Real src_O = src_amp_total * (gO + tailO) * dd * ramp * w_phi;
          Real src_S = src_amp_total * (gS + tailS) * dd * ramp * w_phi;

          // Apply to bulk and scalars
          u(IDN,      k,j,i) += dt * (src_O + src_S);
          u(NHYDRO+0, k,j,i) += dt * f_O * src_O;
          u(NHYDRO+1, k,j,i) += dt * f_S * src_S;

#if NON_BAROTROPIC_EOS
          // Pickup energy per unit mass
          Real dE = (src_O + src_S) * E_pickup;
          u(IEN,k,j,i) += dt * dE;
#endif

          // Pickup momentum toward corotation
          Real rho    = std::max(rho_floor, u(IDN,k,j,i));
          Real vphi   = u(IM3,k,j,i) / rho;
          Real v_corot = u_corot0 * r; // if you added lagged corotation, mirror it here
          Real dm     = dt * (src_O + src_S);
          Real relax  = std::min(1.0, dt/tau_u) * ramp;
          u(IM3,k,j,i) += dm * ((1.0 - relax) * vphi + relax * v_corot);
        }

        // --- Drag toward corotation (optional global) ---
        if (enable_drag) {
          Real rho   = std::max(rho_floor, u(IDN,k,j,i));
          Real vphi  = u(IM3,k,j,i) / rho;
          Real u_corot = u_corot0 * r;
          Real w_drag  = std::exp(-SQR((r - s_io) / (2.0*src_sigma)));
          Real dvphi   = (u_corot - vphi) * (dt/tau_u) * w_drag * ramp;
          u(IM3,k,j,i) += rho * dvphi;
        }

        // --- Gravity taper ---
        if (enable_gravity) {
          Real w_g = 0.0;
          if (r >= s_io + g_taper_width) w_g = 1.0;
          else if (r > s_io) {
            Real xi = (r - s_io)/g_taper_width;
            w_g = 0.5*(1.0 - std::cos(M_PI*xi));
          }
          Real g_r = -GM_custom/(r*r) * w_g;
          Real rho = u(IDN,k,j,i);
          if (rho > 1e-12) u(IM1,k,j,i) += dt * rho * g_r;
        }

        // --- Losses ---
        if (enable_losses) {
          Real xiL = (r <= s_io) ? 0.0 : std::min(1.0,(r - s_io)/1.2);
          Real wL  = 0.5*(1.0 - std::cos(M_PI*xiL));
          Real tau_rho_loc = tau_loss_rho*(1.0 - wL) + 7000.0*wL;

          u(IDN,k,j,i)      -= dt * u(IDN,k,j,i)      / tau_rho_loc;
          u(NHYDRO+0,k,j,i) -= dt * u(NHYDRO+0,k,j,i) / (tau_rho_loc * tau_loss_O / tau_loss_rho);
          u(NHYDRO+1,k,j,i) -= dt * u(NHYDRO+1,k,j,i) / (tau_rho_loc * tau_loss_S / tau_loss_rho);

          u(IDN,      k,j,i) = std::max(rho_floor, u(IDN,      k,j,i));
          u(NHYDRO+0, k,j,i) = std::max(r_floor,   u(NHYDRO+0, k,j,i));
          u(NHYDRO+1, k,j,i) = std::max(r_floor,   u(NHYDRO+1, k,j,i));
        }
      }
    }
  }

  // --- Diffusion limiter ---
  if (enable_diffusion && D_r > 0.0) {
    for (int k = pmb->ks; k <= pmb->ke; ++k) {
      for (int j = pmb->js; j <= pmb->je; ++j) {
        for (int i = pmb->is+1; i <= pmb->ie-1; ++i) {
          Real rC   = coord->x1v(i);
          Real rL   = coord->x1v(i-1);
          Real rR   = coord->x1v(i+1);
          Real drL  = coord->dx1f(i);
          Real drR  = coord->dx1f(i+1);
          Real denom = rC*rC*(0.5*(drL+drR));

          auto lap_update = [&](int var){
            Real qL = u(var,k,j,i-1), qC = u(var,k,j,i), qR = u(var,k,j,i+1);
            Real dql = (qC - qL)/drL, dqr = (qR - qC)/drR;
            auto minmod = [](Real a, Real b){ return (a*b<=0.0) ? 0.0 : ((std::abs(a) < std::abs(b)) ? a : b); };
            Real dq = minmod(dql, dqr);
            Real fluxL = rL*rL*dq, fluxR = rR*rR*dq;
            Real lap = (fluxR - fluxL)/denom;
            u(var,k,j,i) += dt * D_r * lap;
          };
          lap_update(IDN);
          lap_update(NHYDRO+0);
          lap_update(NHYDRO+1);

          u(IDN,      k,j,i) = std::max(rho_floor, u(IDN,      k,j,i));
          u(NHYDRO+0, k,j,i) = std::max(r_floor,   u(NHYDRO+0, k,j,i));
          u(NHYDRO+1, k,j,i) = std::max(r_floor,   u(NHYDRO+1, k,j,i));
        }
      }
    }
  }

  // --- Diagnostics (species + mass + energy + B) ---
  if (pmb->gid == 0 && pmb->lid == 0 && time >= 0.0) {
    Real total_rho=0, total_O=0, total_S=0;
    Real total_E=0, total_B=0;
    Real max_vA=0, max_cf=0;
    Real flux_rho=0;

    #pragma omp parallel for reduction(+:total_rho,total_O,total_S,total_E,total_B,flux_rho) reduction(max:max_vA,max_cf) collapse(3)
    for (int k = pmb->ks; k <= pmb->ke; ++k) {
      for (int j = pmb->js; j <= pmb->je; ++j) {
        for (int i = pmb->is; i <= pmb->ie; ++i) {
          Real r   = coord->x1v(i);
          Real dr  = coord->dx1v(i);
          Real vol = r*r*dr;

          Real rho = u(IDN,k,j,i);
          Real O   = u(NHYDRO+0,k,j,i);
          Real S   = u(NHYDRO+1,k,j,i);
          total_rho += rho * vol;
          total_O   += O   * vol;
          total_S   += S   * vol;

#if NON_BAROTROPIC_EOS
          total_E += u(IEN,k,j,i) * vol;
#endif

#ifdef MAGNETIC_FIELDS
          Real Bmag = std::sqrt(SQR(bcc(IB1,k,j,i)) + SQR(bcc(IB2,k,j,i)) + SQR(bcc(IB3,k,j,i)));
          total_B += Bmag * vol;
          if (rho > 1e-12) {
            Real vA = Bmag / std::sqrt(rho);
            max_vA = std::max(max_vA, vA);
          }
#endif

          Real cs = std::sqrt(peos->GetGamma() * prim(IPR,k,j,i) / rho);
#ifdef MAGNETIC_FIELDS
          Real cf = std::sqrt(cs*cs + SQR(Bmag / std::sqrt(rho)));
#else
          Real cf = cs;
#endif
          max_cf = std::max(max_cf, cf);

          Real ur = u(IM1,k,j,i) / rho;
          flux_rho += rho * ur * vol;
        }
      }
    }

    std::cout << "[diag] t=" << time
              << " M_rho=" << total_rho
              << " M_O+=" << total_O
              << " M_S+=" << total_S
#if NON_BAROTROPIC_EOS
              << " E_tot=" << total_E
#endif
#ifdef MAGNETIC_FIELDS
              << " B_int=" << total_B
              << " max_vA=" << max_vA
#endif
              << " max_cf=" << max_cf
              << " flux_rho=" << flux_rho
              << std::endl;
  }
}
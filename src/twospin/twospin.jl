"""
twospin.jl - common definitions for the twospin directory derived from spin.jl
"""

WDIR = joinpath(@__DIR__, "../../")
include(joinpath(WDIR, "src", "qocexperiments.jl"))

using CSV
using DataFrames
using LinearAlgebra
using Interpolations

# paths
const MM_OUT_PATH = abspath(joinpath(WDIR, "out", "mm"))

chi_minus_df = CSV.read(joinpath(WDIR, "src/twospin/chi_minus_values.csv"), DataFrame, header=false)
gs_minus_expect_df = CSV.read(joinpath(WDIR, "src/twospin/gs_minus_expect_values.csv"), DataFrame, header=false)
J_eff_vals_df = CSV.read(joinpath(WDIR, "src/twospin/J_eff_values.csv"), DataFrame, header=false)
x_vals_df = CSV.read(joinpath(WDIR, "src/twospin/flux_vals.csv"), DataFrame, header=false)
chi_minus_vals = Matrix{Float64}(chi_minus_df)[:,1] #convert(Matrix{Float64}, chi_minus_df)[:, 1]
gs_minus_expect_vals = Matrix{Float64}(gs_minus_expect_df)[:, 1]
J_eff_vals = Matrix{Float64}(J_eff_vals_df)[:, 1]
x_vals = Matrix{Float64}(x_vals_df)[:, 1]

J_eff_spline_itp = extrapolate(interpolate((x_vals,), J_eff_vals,
                                Gridded(Linear())), Periodic())
chi_minus_spline_itp = extrapolate(interpolate((x_vals,), chi_minus_vals,
                                   Gridded(Linear())), Periodic())
gs_minus_spline_itp = extrapolate(interpolate((x_vals,), gs_minus_expect_vals,
                                  Gridded(Linear())), Periodic())

# types
@enum GateType begin
    zzpiby2 = 1
    yypiby2 = 2
    xxpiby2 = 3
    xipiby2 = 4
    ixpiby2 = 5
    yipiby2 = 6
    iypiby2 = 7
    zipiby2 = 8
    izpiby2 = 9
    cnot = 10
    iswap = 11
end

# simulation constants
const DT_PREF = 1e-1

const HDIM_TWOSPIN = 4
const HDIM_TWOSPIN_ISO = HDIM_TWOSPIN * 2
const HDIM_TWOSPIN_VISO = HDIM_TWOSPIN^2
const HDIM_TWOSPIN_VISO_ISO = HDIM_TWOSPIN_VISO * 2

TWOSPIN_ISO_1 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
TWOSPIN_ISO_2 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
TWOSPIN_ISO_3 = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
TWOSPIN_ISO_4 = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]

const PAULI_IDENTITY = [1 0;
                        0 1]
const SIGMAX = [0 1;
                1 0]
const SIGMAZ = [1 0;
                0 -1]
const ZPIBY2_ = [1-1im 0;
                0 1+1im] / sqrt(2)
const YPIBY2_ = [1 -1;
                 1  1] / sqrt(2)
const XPIBY2_ = [1 -1im;
                -1im 1] / sqrt(2)
# Operators for two coupled spins

const WQ_1 = 2π * 0.0072 #0.070
const WQ_2 = 2π * 0.0085 #0.078

const XX = kron(SIGMAX, SIGMAX)
const ZZ = kron(SIGMAZ, SIGMAZ)
const IZ = kron(PAULI_IDENTITY, SIGMAZ)
const ZI = kron(SIGMAZ, PAULI_IDENTITY)
const IX = kron(PAULI_IDENTITY, SIGMAX)
const XI = kron(SIGMAX, PAULI_IDENTITY)
const CNOT = [1 0 0 0;
              0 1 0 0;
              0 0 0 1;
              0 0 1 0]
const sqrt2 = sqrt(2.0)
const iSWAP = [1    0    0 0;
               0    0 -1im 0;
               0 -1im    0 0;
               0    0    0 1]

const XX_ISO = get_mat_iso(XX)
const ZZ_ISO = get_mat_iso(ZZ)
const IZ_ISO = get_mat_iso(IZ)
const ZI_ISO = get_mat_iso(ZI)
const IX_ISO = get_mat_iso(IX)
const XI_ISO = get_mat_iso(XI)
const CNOT_ISO = get_mat_iso(CNOT)
const iSWAP_ISO = get_mat_iso(iSWAP)
const NEGI_H0_TWOSPIN_ISO = get_mat_iso(-1im * WQ_1 * ZI / 2
                                        -1im * WQ_2 * IZ / 2)
const NEGI_H1_TWOSPIN_ISO = get_mat_iso(-1im * 2π * XX)

function J_eff(flux_c)
    return J_eff_spline_itp(flux_c)
end
# two qubit gates

ZZPIBY2_ = kron(ZPIBY2_, ZPIBY2_)
XXPIBY2_ = kron(XPIBY2_, XPIBY2_)
YYPIBY2_ = kron(YPIBY2_, YPIBY2_)
XIPIBY2_ = kron(XPIBY2_, PAULI_IDENTITY)
IXPIBY2_ = kron(PAULI_IDENTITY, XPIBY2_)
YIPIBY2_ = kron(YPIBY2_, PAULI_IDENTITY)
IYPIBY2_ = kron(PAULI_IDENTITY, YPIBY2_)
ZIPIBY2_ = kron(ZPIBY2_, PAULI_IDENTITY)
IZPIBY2_ = kron(PAULI_IDENTITY, ZPIBY2_)

const ZZPIBY2_ISO = get_mat_iso(ZZPIBY2_)
const XXPIBY2_ISO = get_mat_iso(XXPIBY2_)
const YYPIBY2_ISO = get_mat_iso(YYPIBY2_)

const ZZPIBY2_ISO_1 = get_vec_iso(ZZPIBY2_[:,1])
const ZZPIBY2_ISO_2 = get_vec_iso(ZZPIBY2_[:,2])
const ZZPIBY2_ISO_3 = get_vec_iso(ZZPIBY2_[:,3])
const ZZPIBY2_ISO_4 = get_vec_iso(ZZPIBY2_[:,4])

const XXPIBY2_ISO_1 = get_vec_iso(XXPIBY2_[:,1])
const XXPIBY2_ISO_2 = get_vec_iso(XXPIBY2_[:,2])
const XXPIBY2_ISO_3 = get_vec_iso(XXPIBY2_[:,3])
const XXPIBY2_ISO_4 = get_vec_iso(XXPIBY2_[:,4])

const YYPIBY2_ISO_1 = get_vec_iso(YYPIBY2_[:,1])
const YYPIBY2_ISO_2 = get_vec_iso(YYPIBY2_[:,2])
const YYPIBY2_ISO_3 = get_vec_iso(YYPIBY2_[:,3])
const YYPIBY2_ISO_4 = get_vec_iso(YYPIBY2_[:,4])

const CNOT_ISO_1 = get_vec_iso(CNOT[:,1])
const CNOT_ISO_2 = get_vec_iso(CNOT[:,2])
const CNOT_ISO_3 = get_vec_iso(CNOT[:,3])
const CNOT_ISO_4 = get_vec_iso(CNOT[:,4])

const iSWAP_ISO_1 = get_vec_iso(iSWAP[:,1])
const iSWAP_ISO_2 = get_vec_iso(iSWAP[:,2])
const iSWAP_ISO_3 = get_vec_iso(iSWAP[:,3])
const iSWAP_ISO_4 = get_vec_iso(iSWAP[:,4])

const XIPIBY2_ISO_1 = get_vec_iso(XIPIBY2_[:,1])
const XIPIBY2_ISO_2 = get_vec_iso(XIPIBY2_[:,2])
const XIPIBY2_ISO_3 = get_vec_iso(XIPIBY2_[:,3])
const XIPIBY2_ISO_4 = get_vec_iso(XIPIBY2_[:,4])

const IXPIBY2_ISO_1 = get_vec_iso(IXPIBY2_[:,1])
const IXPIBY2_ISO_2 = get_vec_iso(IXPIBY2_[:,2])
const IXPIBY2_ISO_3 = get_vec_iso(IXPIBY2_[:,3])
const IXPIBY2_ISO_4 = get_vec_iso(IXPIBY2_[:,4])

const YIPIBY2_ISO_1 = get_vec_iso(YIPIBY2_[:,1])
const YIPIBY2_ISO_2 = get_vec_iso(YIPIBY2_[:,2])
const YIPIBY2_ISO_3 = get_vec_iso(YIPIBY2_[:,3])
const YIPIBY2_ISO_4 = get_vec_iso(YIPIBY2_[:,4])

const IYPIBY2_ISO_1 = get_vec_iso(IYPIBY2_[:,1])
const IYPIBY2_ISO_2 = get_vec_iso(IYPIBY2_[:,2])
const IYPIBY2_ISO_3 = get_vec_iso(IYPIBY2_[:,3])
const IYPIBY2_ISO_4 = get_vec_iso(IYPIBY2_[:,4])

const ZIPIBY2_ISO_1 = get_vec_iso(ZIPIBY2_[:,1])
const ZIPIBY2_ISO_2 = get_vec_iso(ZIPIBY2_[:,2])
const ZIPIBY2_ISO_3 = get_vec_iso(ZIPIBY2_[:,3])
const ZIPIBY2_ISO_4 = get_vec_iso(ZIPIBY2_[:,4])

const IZPIBY2_ISO_1 = get_vec_iso(IZPIBY2_[:,1])
const IZPIBY2_ISO_2 = get_vec_iso(IZPIBY2_[:,2])
const IZPIBY2_ISO_3 = get_vec_iso(IZPIBY2_[:,3])
const IZPIBY2_ISO_4 = get_vec_iso(IZPIBY2_[:,4])

function target_states(gate_type)
    if gate_type == xxpiby2
        target_state1 = XXPIBY2_ISO_1
        target_state2 = XXPIBY2_ISO_2
        target_state3 = XXPIBY2_ISO_3
        target_state4 = XXPIBY2_ISO_4
    elseif gate_type == yypiby2
        target_state1 = YYPIBY2_ISO_1
        target_state2 = YYPIBY2_ISO_2
        target_state3 = YYPIBY2_ISO_3
        target_state4 = YYPIBY2_ISO_4
    elseif gate_type == zzpiby2
        target_state1 = ZZPIBY2_ISO_1
        target_state2 = ZZPIBY2_ISO_2
        target_state3 = ZZPIBY2_ISO_3
        target_state4 = ZZPIBY2_ISO_4
    elseif gate_type == xipiby2
        target_state1 = XIPIBY2_ISO_1
        target_state2 = XIPIBY2_ISO_2
        target_state3 = XIPIBY2_ISO_3
        target_state4 = XIPIBY2_ISO_4
    elseif gate_type == ixpiby2
        target_state1 = IXPIBY2_ISO_1
        target_state2 = IXPIBY2_ISO_2
        target_state3 = IXPIBY2_ISO_3
        target_state4 = IXPIBY2_ISO_4
    elseif gate_type == yipiby2
        target_state1 = YIPIBY2_ISO_1
        target_state2 = YIPIBY2_ISO_2
        target_state3 = YIPIBY2_ISO_3
        target_state4 = YIPIBY2_ISO_4
    elseif gate_type == iypiby2
        target_state1 = IYPIBY2_ISO_1
        target_state2 = IYPIBY2_ISO_2
        target_state3 = IYPIBY2_ISO_3
        target_state4 = IYPIBY2_ISO_4
    elseif gate_type == zipiby2
        target_state1 = ZIPIBY2_ISO_1
        target_state2 = ZIPIBY2_ISO_2
        target_state3 = ZIPIBY2_ISO_3
        target_state4 = ZIPIBY2_ISO_4
    elseif gate_type == izpiby2
        target_state1 = IZPIBY2_ISO_1
        target_state2 = IZPIBY2_ISO_2
        target_state3 = IZPIBY2_ISO_3
        target_state4 = IZPIBY2_ISO_4
    elseif gate_type == cnot
        target_state1 = CNOT_ISO_1
        target_state2 = CNOT_ISO_2
        target_state3 = CNOT_ISO_3
        target_state4 = CNOT_ISO_4
    elseif gate_type == iswap
        target_state1 = iSWAP_ISO_1
        target_state2 = iSWAP_ISO_2
        target_state3 = iSWAP_ISO_3
        target_state4 = iSWAP_ISO_4
    end
    return (target_state1, target_state2, target_state3, target_state4)
end

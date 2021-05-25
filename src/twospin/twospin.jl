"""
twospin.jl - common definitions for the twospin directory derived from spin.jl
"""

WDIR = joinpath(@__DIR__, "../../")
include(joinpath(WDIR, "src", "qocexperiments.jl"))

using CSV
using DataFrames
using LinearAlgebra
using Interpolations
using Altro

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

@inline Base.size(model::AbstractModel) = model.n, model.m
# vector and matrix constructors (use CPU arrays)
@inline M(mat_) = mat_
@inline Md(mat_) = mat_
@inline V(vec_) = vec_

function initialize_two_spin(model, gate_type, evolution_time, dt,
                             time_optimal, qs, initial_pulse; deriv=false)
    n, m = size(model)
    t0 = 0.
    dt_max = 2 * dt
    dt_min = dt / 1e1
    sqrt_dt_max = sqrt(dt_max)
    sqrt_dt_min = sqrt(dt_min)

    x0 = zeros(n)
    x0[model.state1_idx] = TWOSPIN_ISO_1
    x0[model.state2_idx] = TWOSPIN_ISO_2
    x0[model.state3_idx] = TWOSPIN_ISO_3
    x0[model.state4_idx] = TWOSPIN_ISO_4
    x0 = V(x0)

    # final state
    (target_state1, target_state2,
     target_state3, target_state4) = target_states(gate_type)
    xf = zeros(n)
    xf[model.state1_idx] = target_state1
    xf[model.state2_idx] = target_state2
    xf[model.state3_idx] = target_state3
    xf[model.state4_idx] = target_state4
    xf = V(xf)

    # bound constraints
    x_max = fill(Inf, n)
    x_max_boundary = fill(Inf, n)
    x_min = fill(-Inf, n)
    x_min_boundary = fill(-Inf, n)
    u_max = fill(Inf, m)
    u_max_boundary = fill(Inf, m)
    u_min = fill(-Inf, m)
    u_min_boundary = fill(-Inf, m)
    # constrain the control amplitudes
    x_max[model.controls_idx[1]] = 0.5
    x_min[model.controls_idx[1]] = -0.5
    # control amplitudes go to zero at boundary
    x_max_boundary[model.controls_idx] .= 0
    x_min_boundary[model.controls_idx] .= 0
    # constraint dt
    if time_optimal
        u_max[model.dt_idx] .= sqrt_dt_max
        u_max_boundary[model.dt_idx] .= sqrt_dt_max
        u_min[model.dt_idx] .= sqrt_dt_min
        u_min_boundary[model.dt_idx] .= sqrt_dt_min
    end
    # vectorize
    x_max = V(x_max)
    x_max_boundary = V(x_max_boundary)
    x_min = V(x_min)
    x_min_boundary = V(x_min_boundary)
    u_max = V(u_max)
    u_max_boundary = V(u_max_boundary)
    u_min = V(u_min)
    u_min_boundary = V(u_min_boundary)

    # initial trajectory
    N = Int(floor(evolution_time / dt)) + 1
    X0 = [V(zeros(n)) for k = 1:N]
    X0[1] .= x0
    if isnothing(initial_pulse)
        U0 = [V([
            fill(1e-4, 1);
            fill(dt, time_optimal ? 1 : 0);
        ]) for k = 1:N-1]
        ts = V(zeros(N))
        ts[1] = t0
        for k = 1:N-1
            ts[k + 1] = ts[k] + dt
        end
    else
        U0_, ts = grab_controls(initial_pulse)
        U0 = V(vec([[u0_elem] for u0_elem in U0_]))
    end
    for k = 1:N-1
        discrete_dynamics!(X0[k + 1], EXP, model, X0[k], U0[k], ts[k], dt)
    end

    # cost function
    Q = zeros(n)
    Q[model.state1_idx] .= qs[1]
    Q[model.state2_idx] .= qs[1]
    Q[model.state3_idx] .= qs[1]
    Q[model.state4_idx] .= qs[1]
    Q[model.intcontrols_idx] .= qs[2]
    Q[model.controls_idx] .= qs[3]
    Q[model.dcontrols_idx] .= qs[4]
    if deriv == true
        Q[model.dstate1_idx] .= qs[5]
    end
    Q = Diagonal(V(Q))
    Qf = Q .* N
    # Qf = Q
    R = zeros(m)
    R[model.d2controls_idx] .= qs[6]
    if time_optimal
        R[model.dt_idx] .= qs[7]
    end
    R = Diagonal(V(R))
    objective = LQRObjective(Q, Qf, R, xf, n, m, N, M, V)

    # create constraints
    control_amp = BoundConstraint(x_max, x_min, u_max, u_min, n, m, M, V)
    control_amp_boundary = BoundConstraint(x_max_boundary, x_min_boundary,
                                           u_max_boundary, u_min_boundary, n, m, M, V)
    target_astate_constraint = GoalConstraint(xf, V([model.state1_idx; model.state2_idx;
                                                     model.state3_idx; model.state4_idx;]),
                                              n, m, M, V)
    nidxs = [model.state1_idx, model.state2_idx,
             model.state3_idx, model.state4_idx]
    nc_states = [NormConstraint(EQUALITY, STATE, nidx,
                                  1., n, m, M, V) for nidx in nidxs]
    # add constraints
    constraints = ConstraintList(n, m, N, M, V)
    add_constraint!(constraints, control_amp, V(2:N-2))
    add_constraint!(constraints, control_amp_boundary, V(N-1:N-1))
    add_constraint!(constraints, target_astate_constraint, V(N:N))
    for nc_state in nc_states
        add_constraint!(constraints, nc_state, V(2:N-1))
    end
    return objective, constraints, X0, U0, ts, N
end

function post_process(solver, model, time_optimal, dt, ts, evolution_time, qs)
    acontrols_raw = Altro.controls(solver)
    acontrols_arr = permutedims(reduce(hcat, map(Array, acontrols_raw)), [2, 1])
    astates_raw = Altro.states(solver)
    astates_arr = permutedims(reduce(hcat, map(Array, astates_raw)), [2, 1])
    state1_idx_arr = Array(model.state1_idx)
    state2_idx_arr = Array(model.state2_idx)
    state3_idx_arr = Array(model.state3_idx)
    state4_idx_arr = Array(model.state4_idx)
    controls_idx_arr = Array(model.controls_idx)
    dcontrols_idx_arr = Array(model.dcontrols_idx)
    d2controls_idx_arr = Array(model.d2controls_idx)
    dt_idx_arr = Array(model.dt_idx)
    max_v, max_v_info = Altro.max_violation_info(solver)
    iterations_ = Altro.iterations(solver)
    if time_optimal
        ts = cumsum(map(x -> x^2, acontrols_arr[:,model.dt_idx[1]]))
    end

    result = Dict(
        "acontrols" => acontrols_arr,
        "astates" => astates_arr,
        "dt" => dt,
        "ts" => ts,
        "state1_idx" => state1_idx_arr,
        "state2_idx" => state2_idx_arr,
        "state3_idx" => state3_idx_arr,
        "state4_idx" => state4_idx_arr,
        "controls_idx" => controls_idx_arr,
        "dcontrols_idx" => dcontrols_idx_arr,
        "d2controls_idx" => d2controls_idx_arr,
        "dt_idx" => dt_idx_arr,
        "evolution_time" => evolution_time,
        "max_v" => max_v,
        "max_v_info" => max_v_info,
        "qs" => qs,
        "iterations" => iterations_,
        "time_optimal" => Integer(time_optimal),
        "hdim_iso" => HDIM_TWOSPIN_ISO,
        "save_type" => Int(jl)
    )
    return result
end

function save_to_file!(result, EXPERIMENT_NAME, SAVE_PATH)
    save_file_path = generate_file_path("h5", EXPERIMENT_NAME, SAVE_PATH)
    println("Saving this optimization to $(save_file_path)")
    h5open(save_file_path, "cw") do save_file
        for key in keys(result)
            write(save_file, key, result[key])
        end
    end
    result["save_file_path"] = save_file_path
end

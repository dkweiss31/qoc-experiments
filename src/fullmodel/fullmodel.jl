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

@inline Base.size(model::AbstractModel) = model.n, model.m
# vector and matrix constructors (use CPU arrays)
@inline M(mat_) = mat_
@inline Md(mat_) = mat_
@inline V(vec_) = vec_

# paths
pathtodatafiles = "src/fullmodel/"
H_0_df = CSV.read(joinpath(WDIR, pathtodatafiles, "H_0_np.csv"), DataFrame, header=false)

H_a_r_df = CSV.read(joinpath(WDIR, pathtodatafiles, "H_a_np_r.csv"), DataFrame, header=false)
H_a_i_df = CSV.read(joinpath(WDIR, pathtodatafiles, "H_a_np_i.csv"), DataFrame, header=false)

H_b_r_df = CSV.read(joinpath(WDIR, pathtodatafiles, "H_b_np_r.csv"), DataFrame, header=false)
H_b_i_df = CSV.read(joinpath(WDIR, pathtodatafiles, "H_b_np_i.csv"), DataFrame, header=false)

H_c_lin_r_df = CSV.read(joinpath(WDIR, pathtodatafiles, "H_c_lin_np_r.csv"), DataFrame, header=false)
H_c_lin_i_df = CSV.read(joinpath(WDIR, pathtodatafiles, "H_c_lin_np_i.csv"), DataFrame, header=false)

H_c_cos_r_df = CSV.read(joinpath(WDIR, pathtodatafiles, "H_c_cos_np_r.csv"), DataFrame, header=false)
H_c_cos_i_df = CSV.read(joinpath(WDIR, pathtodatafiles, "H_c_cos_np_i.csv"), DataFrame, header=false)

H_c_sin_r_df = CSV.read(joinpath(WDIR, pathtodatafiles, "H_c_sin_np_r.csv"), DataFrame, header=false)
H_c_sin_i_df = CSV.read(joinpath(WDIR, pathtodatafiles, "H_c_sin_np_i.csv"), DataFrame, header=false)

const H_0 = Matrix{Float64}(H_0_df)
const NEGI_H_0_TWOSPIN_ISO = get_mat_iso(-1im * H_0)

const H_a_r = Matrix{Float64}(H_a_r_df)
const H_a_i = Matrix{Float64}(H_a_i_df)
const H_a = H_a_r + H_a_i .* 1im
const NEGI_H_a_TWOSPIN_ISO = get_mat_iso(-1im * H_a)

const H_b_r = Matrix{Float64}(H_b_r_df)
const H_b_i = Matrix{Float64}(H_b_i_df)
const H_b = H_b_r + H_b_i .* 1im
const NEGI_H_b_TWOSPIN_ISO = get_mat_iso(-1im * H_b)

const H_c_lin_r = Matrix{Float64}(H_c_lin_r_df)
const H_c_lin_i = Matrix{Float64}(H_c_lin_i_df)
const H_c_lin = H_c_lin_r + H_c_lin_i .* 1im
const NEGI_H_c_lin_TWOSPIN_ISO = get_mat_iso(-1im * H_c_lin)

const H_c_cos_r = Matrix{Float64}(H_c_cos_r_df)
const H_c_cos_i = Matrix{Float64}(H_c_cos_i_df)
const H_c_cos = H_c_cos_r + H_c_cos_i .* 1im
const NEGI_H_c_cos_TWOSPIN_ISO = get_mat_iso(-1im * H_c_cos)

const H_c_sin_r = Matrix{Float64}(H_c_sin_r_df)
const H_c_sin_i = Matrix{Float64}(H_c_sin_i_df)
const H_c_sin = H_c_sin_r + H_c_sin_i .* 1im
const NEGI_H_c_sin_TWOSPIN_ISO = get_mat_iso(-1im * H_c_sin)

# types
@enum GateType begin
    cnot = 1
    iswap = 2
    sqrtiswap = 3
    cphase = 4
end

const DT_PREF = 1e-1

const STATE_COUNT = size(H_0)[1]
const STATE_COUNT_ISO = STATE_COUNT * 2

# simulation constants

full_init_iso_list = zeros((STATE_COUNT, STATE_COUNT_ISO))
for state_idx in 1:STATE_COUNT
    full_init_iso = fill(0.0, STATE_COUNT)
    full_init_iso[state_idx] = 1.0
    full_init_iso_list[state_idx, 1:end] = get_vec_iso(full_init_iso)
end
const FULL_INIT_ISO_LIST = full_init_iso_list

# Operators for two coupled spins

const CNOT = [1 0 0 0;
              0 1 0 0;
              0 0 0 1;
              0 0 1 0]
const iSWAP = [1    0    0 0;
               0    0 -1im 0;
               0 -1im    0 0;
               0    0    0 1]
const sqrt2 = sqrt(2.0)
const sqrtiSWAP = [1    0          0       0;
                   0    1/sqrt2 -1im/sqrt2 0;
                   0 -1im/sqrt2    1/sqrt2 0;
                   0    0          0       1]
const CPHASE = [1 0 0 0;
                0 1 0 0;
                0 0 1 0;
                0 0 0 -1]

function enlarge_gate(gate)
    enlarged_gate = Matrix{Complex}(1.0I, STATE_COUNT, STATE_COUNT)
    enlarged_gate[1:4, 1:4] = gate
    return enlarged_gate
end

const sqrtiSWAPfull = enlarge_gate(sqrtiSWAP)
const iSWAPfull = enlarge_gate(iSWAP)
const CNOTfull = enlarge_gate(CNOT)
const CPHASEfull = enlarge_gate(CPHASE)

const TARGET_DICT = Dict(
            iswap => iSWAPfull,
            sqrtiswap => sqrtiSWAPfull,
            cnot => CNOTfull,
            cphase => CPHASEfull
)

function target_states(gate_type)
    target_gate = TARGET_DICT[gate_type]
    target_states = zeros((STATE_COUNT, STATE_COUNT_ISO))
    for state_idx in 1:STATE_COUNT
        target_states[state_idx, 1:end] = get_vec_iso(target_gate[:,state_idx])
    end
    return target_states
end

function initialize_full_model(model, gate_type, evolution_time, dt,
                             time_optimal, qs, initial_pulse;
                             deriv_H0=false, deriv_H1=false)
    n, m = size(model)
    t0 = 0.
    dt_max = 2 * dt
    dt_min = dt / 1e1
    sqrt_dt_max = sqrt(dt_max)
    sqrt_dt_min = sqrt(dt_min)

    # initial state
    x0 = zeros(n)
    for state_idx in 1:STATE_COUNT
        x0[model.state_idxs[state_idx]] = full_init_iso_list[state_idx, 1:end]
    end
    x0 = V(x0)

    # final state
    target_states_iso_list = target_states(gate_type)
    xf = zeros(n)
    for state_idx in 1:STATE_COUNT
        xf[model.state_idxs[state_idx]] = target_states_iso_list[state_idx, 1:end]
    end
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
    x_max[model.controls_idx[1]] = 0.3
    x_min[model.controls_idx[1]] = -0.3
    x_max[model.controls_idx[2]] = 0.3
    x_min[model.controls_idx[2]] = -0.3
    x_max[model.controls_idx[3]] = 0.5
    x_min[model.controls_idx[3]] = -0.5
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
            fill(1e-4, time_optimal ? m - 1 : m);
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
    for state_idx in 1:4
        Q[model.state_idxs[state_idx]] .= qs[1]
    end
    # don't care where the other states end up
    for state_idx in 5:STATE_COUNT
        Q[model.state_idxs[state_idx]] .= 0.0
    end
    Q[model.intcontrols_idx] .= qs[2]
    Q[model.controls_idx] .= qs[3]
    Q[model.dcontrols_idx] .= qs[4]
    if deriv_H0 == true
        Q[model.dstate1_H0_idx] .= qs[5]
    end
    if deriv_H1 == true
        Q[model.dstate1_H1_idx] .= qs[5]
    end
    Q = Diagonal(V(Q))
    Qf = Q .* N
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
    target_astate_constraint = GoalConstraint(xf, V([model.state_idxs[1]; model.state_idxs[2];
                                                     model.state_idxs[3]; model.state_idxs[4];]),
                                              n, m, M, V)
    nc_states = [NormConstraint(EQUALITY, STATE, nidx,
                                  1., n, m, M, V)
                 for nidx in model.state_idxs]
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
        "hdim_iso" => STATE_COUNT_ISO,
        "save_type" => Int(jl),
        "status" => "$(solver.stats.status)"
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

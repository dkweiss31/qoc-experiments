"""
twospin12.jl - vanilla two spin based on spin13.jl
"""
WDIR = abspath(@__DIR__, "../../")
const EXPERIMENT_META = "fluxonium"
include(joinpath(WDIR, "src", EXPERIMENT_META, EXPERIMENT_META * ".jl"))

using Altro
using HDF5
using LinearAlgebra
using StaticArrays

# paths
const EXPERIMENT_NAME = "fluxonium_deriv"
const SAVE_PATH = abspath(joinpath(WDIR, "out", EXPERIMENT_META, EXPERIMENT_NAME))

struct Model{TH,Tis,Tic,Tids,Tif,Tid} <: AbstractModel
    # problem size
    n::Int
    m::Int
    # problem
    Hs::Vector{TH}
    time_optimal::Bool
    # indices
    state_idxs::Tis
    intcontrols_idx::Tic
    controls_idx::Tic
    dcontrols_idx::Tic
    d2controls_idx::Tic
    dstate1_idx::Tids
    forbidden_idx::Tif
    dt_idx::Tid
end

function Model(M_, Md_, V_, Hs, time_optimal)
    # problem size
    control_count = 1
    state_count = STATE_COUNT
    forbid_count = FORBID_COUNT
    n = (state_count * STATE_COUNT_ISO + 3 * control_count
         + 1 * STATE_COUNT_ISO + forbid_count)
    m = time_optimal ? control_count + 1 : control_count
    # state indices
    state_idxs = [V(STATE_COUNT_ISO * (state_idx - 1)
                  + 1:STATE_COUNT_ISO + STATE_COUNT_ISO * (state_idx - 1))
                  for state_idx in 1:state_count]
    intcontrols_idx = V(state_idxs[end][end] + 1:state_idxs[end][end] + control_count)
    controls_idx = V(intcontrols_idx[end] + 1:intcontrols_idx[end] + control_count)
    dcontrols_idx = V(controls_idx[end] + 1:controls_idx[end] + control_count)
    dstate1_idx = V(dcontrols_idx[end] + 1:dcontrols_idx[end] + STATE_COUNT_ISO)
    forbidden_idx = V(dstate1_idx[end] + 1:dstate1_idx[end] + forbid_count)
    # control indices
    d2controls_idx = V(1:control_count)
    dt_idx = V(d2controls_idx[end] + 1:d2controls_idx[end] + 1)
    # types
    TH = typeof(Hs[1])
    Tis = typeof(state_idxs)
    Tic = typeof(controls_idx)
    Tids = typeof(dstate1_idx)
    Tif = typeof(forbidden_idx)
    Tid = typeof(dt_idx)
    return Model{TH,Tis,Tic,Tids,Tif,Tid}(n, m, Hs, time_optimal, state_idxs,
                                     intcontrols_idx, controls_idx,
                                     dcontrols_idx, d2controls_idx,
                                     dstate1_idx, forbidden_idx, dt_idx)
end

# dynamics
abstract type EXP <: Explicit end

# dynamics
function Altro.discrete_dynamics(::Type{EXP}, model::Model,
                              astate::AbstractVector,
                              acontrol::AbstractVector, time::Real, dt_::Real)
    dt = !model.time_optimal ? dt_ : acontrol[model.dt_idx[1]]^2
    H = dt * (model.Hs[1] + astate[model.controls_idx[1]] * model.Hs[2])
    U = exp(H)
    astate_ = []
    for state_idx in 1:STATE_COUNT
        new_state = U * astate[model.state_idxs[state_idx]]
        append!(astate_, new_state)
    end
    intcontrols = astate[model.intcontrols_idx] + astate[model.controls_idx] .* dt
    controls = astate[model.controls_idx] + astate[model.dcontrols_idx] .* dt
    dcontrols = astate[model.dcontrols_idx] + acontrol[model.d2controls_idx] .* dt
    state1_ = astate[model.state_idxs[1]]
    dstate1_ = astate[model.dstate1_idx]
    dstate1 = U * (dstate1_ + dt * model.Hs[1] * state1_)
    append!(astate_, [intcontrols; controls; dcontrols; dstate1])
    if FORBID_COUNT >= 1
        for state_idx_qubit in 1:2
            overall_forbidden_overlap = 0.0
            for state_idx_forbidden in 1:FORBID_COUNT
                forbidden_overlap = state_overlap_norm(astate[model.state_idxs[state_idx_qubit]],
                                                       FORBIDDEN_STATES[state_idx_forbidden, 1:end])
                overall_forbidden_overlap += forbidden_overlap
            end
            append!(astate_, overall_forbidden_overlap)
        end
    end
    return astate_
end

# main
function run_traj(;gate_type=xpiby2, evolution_time=20., dt=DT_PREF,
                  verbose=true, time_optimal=false,
                  qs=[1e0, 1e0, 1e0, 1e-1, 5e-2, 1e-1, 1e-1, 1e-1],
                  smoke_test=false, save=true, max_iterations=10000, bp_reg_fp=10.,
                  dJ_counter_limit=20, bp_reg_type=:control, projected_newton=true,
                  ilqr_max_iterations=600, initial_pulse=nothing,
                  deriv_H0=false, deriv_H1=false)
    # model configuration
    num_steps = floor(evolution_time / dt)
    evolution_time = num_steps * dt
    Hs = [M(H) for H in (NEGI_H_0_FLUXONIUM_ISO,
                         NEGI_H_CONT_FLUXONIUM_ISO)]
    model = Model(M, Md, V, Hs, time_optimal)
    objective, constraints, X0, U0, ts, N = initialize_fluxonium(model, gate_type, evolution_time, dt,
                                                                time_optimal, qs, initial_pulse,
                                                                deriv_H0=deriv_H0, deriv_H1=deriv_H1)
    # build problem
    prob = Problem(EXP, model, objective, constraints, X0, U0, ts, N, M, Md, V)
    opts = SolverOptions(
        verbose_pn=verbose ? true : false, verbose=verbose ? 2 : 0,
        ilqr_max_iterations=smoke_test ? 1 : ilqr_max_iterations,
        al_max_iterations=smoke_test ? 1 : 30, n_steps=smoke_test ? 1 : 2,
        iterations=max_iterations, bp_reg_fp=bp_reg_fp,
        dJ_counter_limit=dJ_counter_limit, bp_reg_type=bp_reg_type,
        projected_newton=false
    )
    # solve
    solver = ALTROSolver(prob, opts)
    Altro.solve!(solver)
    println("status: $(solver.stats.status)")

    # post-process
    result = post_process(solver, model, time_optimal, dt, ts, evolution_time, qs)

    # save
    if save
        save_to_file!(result, EXPERIMENT_NAME, SAVE_PATH)
    end

    return result
end

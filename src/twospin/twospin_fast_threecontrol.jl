"""
twospin12.jl - vanilla two spin based on spin13.jl
with control of all three fluxes
"""
WDIR = abspath(@__DIR__, "../../")
const EXPERIMENT_META = "twospin"
include(joinpath(WDIR, "src", EXPERIMENT_META, EXPERIMENT_META * ".jl"))

using Altro
using HDF5
using LinearAlgebra
using StaticArrays

# paths
const EXPERIMENT_META = "twospin"
const EXPERIMENT_NAME = "twospinfastthreecontrol"
const SAVE_PATH = abspath(joinpath(WDIR, "out", EXPERIMENT_META, EXPERIMENT_NAME))

struct Model{TH,Tis,Tic,Tid} <: AbstractModel
    # problem size
    n::Int
    m::Int
    # problem
    Hs::Vector{TH}
    time_optimal::Bool
    # indices
    state1_idx::Tis
    state2_idx::Tis
    state3_idx::Tis
    state4_idx::Tis
    intcontrols_idx::Tic
    controls_idx::Tic
    dcontrols_idx::Tic
    d2controls_idx::Tic
    dt_idx::Tid
end

function Model(M_, Md_, V_, Hs, time_optimal)
    # problem size
    control_count = 3
    state_count = HDIM_TWOSPIN
    n = state_count * HDIM_TWOSPIN_ISO + 3 * control_count
    m = time_optimal ? control_count + 1 : control_count
    # state indices
    state1_idx = V(1:HDIM_TWOSPIN_ISO)
    state2_idx = V(state1_idx[end] + 1:state1_idx[end] + HDIM_TWOSPIN_ISO)
    state3_idx = V(state2_idx[end] + 1:state2_idx[end] + HDIM_TWOSPIN_ISO)
    state4_idx = V(state3_idx[end] + 1:state3_idx[end] + HDIM_TWOSPIN_ISO)
    intcontrols_idx = V(state4_idx[end] + 1:state4_idx[end] + control_count)
    controls_idx = V(intcontrols_idx[end] + 1:intcontrols_idx[end] + control_count)
    dcontrols_idx = V(controls_idx[end] + 1:controls_idx[end] + control_count)
    # control indices
    d2controls_idx = V(1:control_count)
    dt_idx = V(d2controls_idx[end] + 1:d2controls_idx[end] + 1)
    # types
    TH = typeof(Hs[1])
    Tis = typeof(state1_idx)
    Tic = typeof(controls_idx)
    Tid = typeof(dt_idx)
    return Model{TH,Tis,Tic,Tid}(n, m, Hs, time_optimal, state1_idx, state2_idx,
                                 state3_idx, state4_idx, intcontrols_idx,
                                 controls_idx, dcontrols_idx, d2controls_idx, dt_idx)
end

# dynamics
abstract type EXP <: Explicit end

# dynamics
function Altro.discrete_dynamics(::Type{EXP}, model::Model,
                              astate::AbstractVector,
                              acontrol::AbstractVector, time::Real, dt_::Real)
    dt = !model.time_optimal ? dt_ : acontrol[model.dt_idx[1]]^2
    H = dt * (model.Hs[1] +
              J_eff(astate[model.controls_idx[1]]+0.28) * model.Hs[2]
              + (ELa * model.controls_idx[2]
                 + J_eff(astate[model.controls_idx[1]]+0.28) * model.controls_idx[3]
                 - 0.5 * ELa * gs_minus_spline_itp(astate[model.controls_idx[1]]+0.28))
                * model.Hs[3] * 2.950
              + (ELb * model.controls_idx[3]
                 + J_eff(astate[model.controls_idx[1]]+0.28) * model.controls_idx[2]
                 + 0.5 * ELb * gs_minus_spline_itp(astate[model.controls_idx[1]]+0.28))
                * model.Hs[4] * 2.911
              )
    U = exp(H)
    state1 = U * astate[model.state1_idx]
    state2 = U * astate[model.state2_idx]
    state3 = U * astate[model.state3_idx]
    state4 = U * astate[model.state4_idx]
    intcontrols = astate[model.intcontrols_idx] + astate[model.controls_idx] .* dt
    controls = astate[model.controls_idx] + astate[model.dcontrols_idx] .* dt
    dcontrols = astate[model.dcontrols_idx] + acontrol[model.d2controls_idx] .* dt
    astate_ = [state1; state2; state3; state4; intcontrols; controls; dcontrols]
    return astate_
end

# main
function run_traj(;gate_type=iswap, evolution_time=50., dt=DT_PREF, verbose=true,
                  time_optimal=false, qs=[1e0, 1e-1, 1e-1, 1e-1, 5e-2, 1e-1, 1e-1], smoke_test=false,
                  save=true, max_iterations=10000, bp_reg_fp=10.,
                  dJ_counter_limit=20, bp_reg_type=:control, projected_newton=true,
                  initial_pulse=nothing)
    # model configuration
    Hs = [M(H) for H in (NEGI_H0_TWOSPIN_ISO, NEGI_H1_TWOSPIN_ISO,
                         NEGI_XI_TWOSPIN_ISO, NEGI_IX_TWOSPIN_ISO)]
    model = Model(M, Md, V, Hs, time_optimal)
    objective, constraints, X0, U0, ts, N = initialize_two_spin(model, gate_type, evolution_time, dt,
                                                                time_optimal, qs, initial_pulse)
    # build problem
    prob = Problem(EXP, model, objective, constraints, X0, U0, ts, N, M, Md, V)
    opts = SolverOptions(
        verbose_pn=verbose ? true : false, verbose=verbose ? 2 : 0,
        ilqr_max_iterations=smoke_test ? 1 : 300,
        al_max_iterations=smoke_test ? 1 : 30, n_steps=smoke_test ? 1 : 2, iterations=max_iterations,
        bp_reg_fp=bp_reg_fp, dJ_counter_limit=dJ_counter_limit, bp_reg_type=bp_reg_type,
        projected_newton=projected_newton,
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

using Distributions
using Random
using LinearAlgebra
using PreallocationTools
using ForwardDiff

### MATLAB generated symbolic functions ###
# TODO: improve stability by looking at small number divisions,
# TODO: add max(0, res) to functions with domain ℜ₊ (positive reals)
# Domain errors, print arguments.
dhrpds1(s1, s2, speed_β, pace_length, td1, td2, strength_hr, σ) =
    -(
        strength_hr *
        exp(
            -(
                (
                ((s1^2 + s2^2)^(1 / 2) +
                 (
                    (s1 - td1 * pace_length * speed_β)^2 +
                    (s2 - td2 * pace_length * speed_β)^2
                )^(1 / 2)
                )^2 - pace_length^2 * speed_β^2)
            )^(1 / 2) / (2 * σ),
        ) *
        (
            (s1^2 + s2^2)^(1 / 2) +
            (
                (s1 - td1 * pace_length * speed_β)^2 + (s2 - td2 * pace_length * speed_β)^2
            )^(1 / 2)
        ) *
        (
            s1 / (s1^2 + s2^2)^(1 / 2) +
            (2 * s1 - 2 * td1 * pace_length * speed_β) / (
                2 *
                (
                    (s1 - td1 * pace_length * speed_β)^2 +
                    (s2 - td2 * pace_length * speed_β)^2
                )^(1 / 2)
            )
        )
    ) / (
        2 *
        σ *
        (
            (
                (s1^2 + s2^2)^(1 / 2) +
                (
                    (s1 - td1 * pace_length * speed_β)^2 +
                    (s2 - td2 * pace_length * speed_β)^2
                )^(1 / 2)
            )^2 - pace_length^2 * speed_β^2)^(1 / 2)
    )
dhrpds2(s1, s2, speed_β, pace_length, td1, td2, strength_hr, σ) =
    -(
        strength_hr *
        exp(
            -(
                (
                    (s1^2 + s2^2)^(1 / 2) +
                    (
                        (s1 - td1 * pace_length * speed_β)^2 +
                        (s2 - td2 * pace_length * speed_β)^2
                    )^(1 / 2)
                )^2 - pace_length^2 * speed_β^2)^(1 / 2) / (2 * σ),
        ) *
        (
            (s1^2 + s2^2)^(1 / 2) +
            (
                (s1 - td1 * pace_length * speed_β)^2 + (s2 - td2 * pace_length * speed_β)^2
            )^(1 / 2)
        ) *
        (
            s2 / (s1^2 + s2^2)^(1 / 2) +
            (2 * s2 - 2 * td2 * pace_length * speed_β) / (
                2 *
                (
                    (s1 - td1 * pace_length * speed_β)^2 +
                    (s2 - td2 * pace_length * speed_β)^2
                )^(1 / 2)
            )
        )
    ) / (
        2 *
        σ *
        (
            (
                (s1^2 + s2^2)^(1 / 2) +
                (
                    (s1 - td1 * pace_length * speed_β)^2 +
                    (s2 - td2 * pace_length * speed_β)^2
                )^(1 / 2)
            )^2 - pace_length^2 * speed_β^2
        )^(1 / 2)
    )

"""
    human_repulsive_effect(strength_hr::Real, s_αβ, speed_β::Real, pace_length::Real, td_β, σ::Real)
The repulsive force exerted onto agent α by agent β.

Arguments
===
+ strength_hr: a scalar value representing the strength of the force,
+ s_αβ: the displacement vector between agents α and β, can also be a tuple
+ speed_β: the speed of agent β,
+ pace_length: a parameter chosen such that speed_β * pace_length ≈ human step length, it is not the step size of time,
+ td_β: the target direction of agent β, can also be a tuple
+ σ: the scalar shape parameter of the force,

Result
===
A tuple
"""
function human_repulsive_effect(
    strength_hr::Real,
    s_αβ,
    speed_β::Real,
    pace_length::Real,
    td_β,
    σ::Real,
)
    -dhrpds1(s_αβ[1], s_αβ[2], speed_β, pace_length, td_β[1], td_β[2], strength_hr, σ),
    -dhrpds2(s_αβ[1], s_αβ[2], speed_β, pace_length, td_β[1], td_β[2], strength_hr, σ)

end

function Dhrp(strength_hr, s1, s2, sigma)
    num = -(strength_hr * exp(-(s1^2 + s2^2)^(1 / 2) / sigma)) / (sigma * (s1^2 + s2^2)^(1 / 2))
    mt = (s1, s2)
    return num .* mt
end

function hre(
    strength_hr::Real,
    s_αβ,
    speed_β::Real,
    pace_length::Real,
    td_β,
    σ::Real,
)
    return .-Dhrp(strength_hr, s_αβ[1], s_αβ[2], σ)
end


dwrpds1(s1, s2, U0, R) =
    -(U0 * s1 * exp(-(s1^2 + s2^2)^(1 / 2) / R)) / (R * (s1^2 + s2^2)^(1 / 2))
dwrpds2(s1, s2, U0, R) =
    -(U0 * s2 * exp(-(s1^2 + s2^2)^(1 / 2) / R)) / (R * (s1^2 + s2^2)^(1 / 2))
wall_repulsive_effect(strength_wr::Real, r_αB, R::Real) =
    -dwrpds1(r_αB[1], r_αB[2], strength_wr, R), -dwrpds2(r_αB[1], r_αB[2], strength_wr, R)

"""
    social_force!(du, u, p, t)
Calculates force for each pedestrian and updates `du` inplace
    Return: `nothing`
"""
function social_force!(du, u, p, t)
    τs,
    pace_length,
    strength_hr,
    shape_hr,
    strength_wr,
    shape_wr,
    φ,
    c,
    target_directions,
    target_speeds,
    F_total,
    vect_of_segments = p

    N = size(du, 2) # Number of agents
    F_total = get_tmp(F_total, u) # Gets preallocated memory of correct type

    for i = 1:N
        # Reset du for each player based on u
        du[1, i] = u[3, i]
        du[2, i] = u[4, i]
        du[3, i] = zero(eltype(u))
        du[4, i] = zero(eltype(u))

        # Main effect 1: They want to reach a certain target destination as comfortably as possible.
        tv = target_speeds[i] .* target_directions[i]
        v = (u[3, i], u[4, i])
        @. F_total = (tv .- v) / τs[i]

        # Main effect 2.1: The motion of a pedestrian α is influenced by other pedestrians.
        # They keep a certain distance away from other pedestrians which depends on the pedestrian density
        # and the target speed.
        s_α = (u[1, i], u[2, i])
        v_α = (u[3, i], u[4, i])
        speed_α = norm(v_α)
        td_α = v_α ./ speed_α
        for j = Iterators.flatten((1:i-1, i+1:N)) # "for j = 1:N, j != i" force j has on i
            Δt = 0.0
            A = strength_hr
            B = shape_hr
            v_β = (u[3, j], u[4, j])
            y_αβ = (v_β .- v_α) .* Δt
            s_β = (u[1, j], u[2, j])
            s_αβ = s_α .- s_β
            s_βα = .-s_αβ

            b_αβ = 0.5 * √((norm(s_αβ) + norm(s_αβ .- y_αβ .* Δt))^2 - (norm(y_αβ .* Δt))^2)
            snorm(v) = all(v .== 0.0) ? Inf : norm(v)
            f_αβ(s_αβ) = A * exp(-b_αβ / B) .* (norm(s_αβ) + norm(s_αβ .- y_αβ)) / 2b_αβ .* (0.5 .* (s_αβ ./ snorm(s_αβ) .+ (s_αβ .- y_αβ) ./ snorm(s_αβ .- y_αβ)))
            # speed_β = norm(v_β)
            # td_β = v_β ./ speed_β
            # hrej = .-hre(strength_hr, s_βα, speed_α, pace_length, td_α, shape_hr)
            # F_total .+= (dot(td_β, hrej) >= norm(hrej) * cos(φ) ? hrej : c .* hrej)
            F_total .+= f_αβ(s_αβ)
        end

        # Main effect 2.2: A pedestrian also keeps a certain distance from borders of buildings, walls,
        # streets, obstacles, etc.
        for segments in vect_of_segments
            min_distance = Inf
            min_closest_point = (0.0, 0.0)
            for segment in segments
                current_closest_point = closest_point(s_α, segment)
                current_distance = norm(s_α .- current_closest_point)
                if min_distance > current_distance
                    min_distance = current_distance
                    min_closest_point = current_closest_point
                end
            end
            displacement = min_closest_point .- s_α
            F_total .-= wall_repulsive_effect(strength_wr, displacement, shape_wr)
        end

        # Update acceleration and velocities
        du[3, i], du[4, i] = F_total
        du[1, i] += du[3, i]
        du[2, i] += du[4, i]
        # Adjust the velocity to reduce if speed is over the agent's maximum speed 
        velocity = (du[1, i], du[2, i])
        speed = norm(velocity)
        maxspeed = 1.3 * target_speeds[i]
        du[1, i], du[2, i] = speed > maxspeed ? velocity .* (maxspeed / speed) : velocity
    end
    return nothing
end


# Previously used functions replaced with their corresponding code to simplify/shorten code.
# Kept here for documentation purposes.
"""
    acceleration_term(v::AbstractVector, tv::AbstractVector, τ::Real)
Calculates the acceleration term from Main Effect 1.

Arguments   
===
+ `v`: the current velocity of the agent,
+ `tv`: target velocity of the agent.
+ `τ`: relaxation time of the agent.

Result
===
A vector.
"""
@inline acceleration_term(v, tv, τ::Real) = (1 / τ) * (tv - v)


"""
    b(s_αβ::Vector, speed_β::Real, pace_length::Real, td_β::Vector)
The semi-minor axis of the elipse formed by the equipotent contours of the
[`human_repulsive_potential`](@ref).

Arguments   
===
+ `s_αβ`: the displacement vector between agents `α` and `β`,
+ `speed_β`: the speed of agent `β`,
+ `pace_length`: a parameter chosen such that `speed_β` * `pace_length` ≈ human step length, it is _not_ the step size of time,
+ `td_β`: the target direction of agent `β`.

Result
===
A scalar
"""
function b(s_αβ, speed_β::Real, pace_length::Real, td_β)
    pace_length_β = speed_β * pace_length
    (1 / 2) * √((norm(s_αβ) + norm(s_αβ .- pace_length_β .* td_β))^2 - pace_length_β^2)
end

"""
    human_repulsive_potential(strength_hr::Real, s_αβ::Vector, speed_β::Real, pace_length::Real, td_β::Vector, σ::Real)
A monotonic decreasing function of b with equipotential lines having the form of an ellipse
that is directed into the direction of motion.

Arguments   
===
+ `strength_hr`: a scalar value representing the human repulsive strength (in paper),
+ `s_αβ`: the displacement vector between agents `α` and `β`,
+ `speed_β`: the speed of agent `β`,
+ `pace_length`: a parameter chosen such that `speed_β` * `pace_length` ≈ human step length, it is _not_ the step size of time,
+ `td_β`: the target direction of agent `β`,
+ `σ`: shape parameter of the human repulsive potential.

Result
===
A vector
"""
function human_repulsive_potential(strength_hr::Real, s_αβ, speed_β::Real, pace_length::Real, td_β, σ::Real)
    strength_hr * exp(-b(s_αβ, speed_β, pace_length, td_β) / σ)
end


function ∇hrp(strength_hr::Real, s_αβ, speed_β::Real, pace_length::Real, td_β, σ::Real)
    ForwardDiff.gradient(s -> human_repulsive_potential(strength_hr, s, speed_β, pace_length, td_β, σ), [s_αβ...])
end

"""
    direction_dependent_weights(e, f, φ, c)

Arguments
===
+ e: a force,
+ f: another force,
+ φ: 2φ is the field of view of the agent,
+ c: the factor if the interaction is precieved outside the FoV given by φ,

Result
===
A 2-length object
"""
@inline direction_dependent_weights(e, f, φ, c) = dot(e, f) >= norm(f) * cos(φ) ? 1 : c
# Not differentiable everywhere, not the issue though

"""
    weighted_repulsive_effect(e, hre, φ, c)
The formulas for repulsive effects only hold for situations that are perceived 
in the target direction `td` of motion. Situations located behind a pedestrian
will have a weaker influence c with 0 < c < 1. In order to take this effect of
perception (i.e. of the effective angle 2ϕ of sight) into account we have to
introduce the direction dependent weights)

Arguments
===
+ td: the target direction,
+ hre: the human_repulsive_effect calculated by [`human_repulsive_effect`](@ref),
+ ϕ: the angle made from the direction of motion i.e. the field of view is 2ϕ,
+ c: a constant damping term for events happening behind the agent

Result
===
A 2-length object
"""
@inline weighted_repulsive_effect(td, hre, φ, c) =
    direction_dependent_weights(td, hre, φ, c) .* hre

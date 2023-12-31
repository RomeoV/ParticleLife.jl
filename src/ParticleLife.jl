module ParticleLife
using Agents
using Random
import StatsBase: middle
import StaticArraysCore: SVector, MVector
using FLoops
using FoldsThreads
using DataStructures: OrderedDict
import LinearAlgebra: norm

export agent_step!, make_model, color_sym

greet() = print("Hello World!")

abstract type ParticleColor end
struct Red <: ParticleColor end
struct Green <: ParticleColor end
struct Yellow <: ParticleColor end
struct Cyan <: ParticleColor end
struct Orange <: ParticleColor end

@agent struct Particle(ContinuousAgent{2, Float64})
    color::ParticleColor
end

function make_model()
    extent::NTuple{2, Float64} = 2 .* (500.0, 500.0);
    space2d = ContinuousSpace(extent;
                              periodic=false,
                              # update_vel! = update_vel!,
                              spacing=5,
                              );
                              # spacing=minimum(extent)/200)
    model = AgentBasedModel(Particle, space2d;
                            agent_step! = agent_step!, model_step! = model_step!,
                            properties=Dict{Symbol, Float64}(
                                            :red_red       =>0.64,
                                            :red_green     =>0.65,
                                            :red_yellow    =>0.76,
                                            :red_cyan      =>0.76,
                                            :green_red     =>0.55,
                                            :green_green   =>-0.32,
                                            :green_yellow  =>-0.41,
                                            :cyan_cyan     =>0.76,
                                            :yellow_red    =>-0.93,
                                            :yellow_green  =>0.86,
                                            :yellow_yellow =>-0.56,
                                            :yellow_cyan   =>-0.56,
                                            :cyan_red      =>-0.93,
                                            :cyan_green    =>0.86,
                                            :cyan_yellow   =>-0.56,
                                            :cyan_cyan     =>0.76,
                                            :time_scale    => 1.0,
                                            :viscosity     => 0.1,
                            ))

    for c in [Red(), Green(), Yellow(), Cyan()]
        for _ in 1:200
            # vel =  SVector(rand() * 400 + 50, rand() * 400 + 50)
            vel =  SVector(0., 0.)
            add_agent!(model, vel, c)
        end
    end
    @assert all(0 <= agent.pos[i] <= extent[i] for agent in allagents(model), i=1:2)
    model
end

color_interact(lhs::ParticleColor,  rhs::ParticleColor, m::ABM) = abmproperties(m)[Symbol(string(color_sym(lhs))*'_'*string(color_sym(rhs)))]
    # color_interact(lhs, rhs)
# color_interact(::Red,  ::Red, m::ABM) = abmproperties(m)[:red_red]
# color_interact(::ParticleColor,  ::ParticleColor) = 0;
# # Green with
# color_interact(::Green,  ::Green)  = -0.32
# color_interact(::Green,  ::Red)    = 0.55;
# color_interact(::Green,  ::Yellow) = -0.41;
# # Red with
# color_interact(::Red,    ::Green)  = 0.65;
# color_interact(::Red,    ::Red)    = 0.64;
# color_interact(::Red,    ::Yellow) = 0.76;
# # Yellow with
# color_interact(::Yellow, ::Red  )  = -0.93;
# color_interact(::Yellow, ::Green)  = 0.86;
# color_interact(::Yellow, ::Yellow) = -0.56;

color_sym(p::Particle) = color_sym(p.color)
color_sym(::ParticleColor) = :black
color_sym(::Green) = :green
color_sym(::Red) = :red
color_sym(::Yellow) = :yellow
color_sym(::Cyan) = :cyan

function model_step!(model)
    total_vel = 0.
    @floop for agent in allagents(model)
    # for agent in allagents(model)
        update_vel!(agent, model)
        @reduce(total_vel += norm(agent.vel))
    end
    total_vel /= length(allagents(model))
    if total_vel > 30.
        model.time_scale /= 1.1
    end
    if model.time_scale < 0.9
        model.time_scale *= 1.01
    elseif model.time_scale > 1.1
        model.time_scale /= 1.01
    end
    @debug model.time_scale
end

agent_step!(agent, model) = move_agent!(agent, model, abmproperties(model)[:time_scale])
function update_vel!(agent::Particle, model::ABM)
    force = sum(
        let g = color_interact(agent.color, other.color, model), d = euclidean_distance(agent, other, model)
            (0 < d < 80 ? (g / d .* (agent.pos - other.pos)) : SVector(0., 0.))
        end
        for other in Agents.nearby_agents(agent, model, 80);
        init = SVector(0., 0.),
    )
    # push away from border
    force += 0.1*(max.(40 .- agent.pos, 0)
                 - max.(40 .- (spacesize(model) - agent.pos), 0))
    # vel = let viscosity = 0.1
    #     vel_ = convert(MVector, middle.(agent.vel, force))
    #     vel_ .*= (1-viscosity)
    #     clamp!(vel_, -400, 400)
    #     return SVector(vel_...)
    # end
    # @show vel
    viscosity = abmproperties(model)[:viscosity]
    # agent.vel = middle.(agent.vel, force) * (1-viscosity)
    agent.vel = agent.vel * (1-viscosity) + force
end

properties=OrderedDict(
    :red_red       => -1:0.1:1,
    :red_green     => -1:0.1:1,
    :red_yellow    => -1:0.1:1,
    :green_red     => -1:0.1:1,
    :green_green   => -1:0.1:1,
    :green_yellow  => -1:0.1:1,
    :yellow_red    => -1:0.1:1,
    :yellow_green  => -1:0.1:1,
    :yellow_yellow => -1:0.1:1,
    :viscosity     => 0.0:0.01:1.0,
)

end # module ParticleLife

module ParticleLife
using Agents
using Random
import StatsBase: middle
import StaticArraysCore: SVector
using FLoops

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
    extent::NTuple{2, Float64} = (500.0, 500.0);
    space2d = ContinuousSpace(extent;
                              periodic=false,
                              # update_vel! = update_vel!,
                              spacing=minimum(extent)/20
                              );
                              # spacing=minimum(extent)/200)
    model = AgentBasedModel(Particle, space2d; agent_step! = agent_step!, model_step! = model_step!)

    for c in [Red(), Green(), Yellow()]
        for _ in 1:200
            # vel =  SVector(rand() * 400 + 50, rand() * 400 + 50)
            vel =  SVector(0., 0.)
            add_agent!(model, vel, c)
        end
    end
    @assert all(0 <= agent.pos[i] <= extent[i] for agent in allagents(model), i=1:2)
    model
end

color_interact(::ParticleColor,  ::ParticleColor) = 0;
color_interact(::Green,  ::Green)  = -0.32;
color_interact(::Green,  ::Red)    = -0.17;
color_interact(::Green,  ::Yellow) =  0.34;
color_interact(::Red,    ::Red)    = -0.1;
color_interact(::Red,    ::Green)  = -0.34;
color_interact(::Yellow, ::Yellow) =  0.15;
color_interact(::Yellow, ::Green)  = -0.2;

color_sym(p::Particle) = color_sym(p.color)
color_sym(::ParticleColor) = :black
color_sym(::Green) = :green
color_sym(::Red) = :red
color_sym(::Yellow) = :yellow

function model_step!(model)
    @floop for agent in allagents(model)
    # for agent in allagents(model)
        update_vel!(agent, model)
    end
end

agent_step!(agent, model) = move_agent!(agent, model, 0.1)
function update_vel!(agent::Particle, model::ABM)
    force = sum(
        let g = color_interact(agent.color, other.color), d = euclidean_distance(agent, other, model)
            g / (d+1e-5) .* (agent.pos - other.pos)
        end
        for other in Agents.nearby_agents(agent, model, 80);
        init = SVector(0., 0.),
    )
    # push away from border
    force += 100*(max.(1 ./ agent.pos .- 1/40, 0)
                 - max.(1 ./(spacesize(model) - agent.pos) .- 1/40, 0))
    agent.vel = middle.(agent.vel, force)
    agent.vel = clamp.(agent.vel, -400, 400)
end

end # module ParticleLife

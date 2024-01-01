module ParticleLife
using Agents
using Random
import StatsBase: middle
import StaticArraysCore: SVector, MVector
using FLoops
using DataStructures: OrderedDict
import LinearAlgebra: norm
using Makie, Makie.Observables
using OnlineStats
import DataStructures: DefaultDict
using ThreadPinning
using TimerOutputs

export agent_step!, make_model, color_sym, run_sim

abstract type ParticleColor end
struct Red <: ParticleColor end
struct Green <: ParticleColor end
struct Yellow <: ParticleColor end
struct Cyan <: ParticleColor end
struct Orange <: ParticleColor end

@agent struct Particle(ContinuousAgent{2, Float64})
    color::ParticleColor
end

function make_model(to=TimerOutput())
    extent::NTuple{2, Float64} = 3 .* (500.0, 500.0);
    space2d = ContinuousSpace(extent;
                              periodic=false,
                              # spacing=20,
                              );
    model = AgentBasedModel(Particle, space2d;
                            agent_step! = (args...) -> (@timeit to "agent_step!" agent_step!(args...)),
                            model_step! = (args...) -> (@timeit to "model_step!" model_step!(args...; to=to)
                            ),
                            properties=push!(Dict{Symbol, Float64}(
                                                lhs_rhs=>rand(range)
                                                for (lhs_rhs, range) in properties),
                                            :time_scale => 1.0)
                            )

    for c in [Red(), Green(), Orange(), Cyan()]
        for _ in 1:750
            vel =  SVector(0., 0.)
            add_agent!(model, vel, c)
        end
    end
    @assert all(0 <= agent.pos[i] <= extent[i] for agent in allagents(model), i=1:2)

    # for tracking fps
    model
end

color_interact(lhs::ParticleColor,  rhs::ParticleColor, m::ABM) =
    abmproperties(m)[Symbol(string(color_sym(lhs))*'_'*string(color_sym(rhs)))]

color_sym(p::Particle) = color_sym(p.color)
color_sym(::ParticleColor) = :black
color_sym(::Green) = :green
color_sym(::Red) = :red
color_sym(::Orange) = :orange
color_sym(::Cyan) = :cyan
color_sym(::Yellow) = :yellow

last_model_step_time::UInt64 = time_ns()
avg_model_step_duration::Observable{Mean{Float64}} = Observable(Mean(weight=HarmonicWeight(30)))

function model_step!(model; to=TimerOutput())
    @timeit to "update_vel! loop" begin
        viscosity::Float64 = abmproperties(model)[:viscosity]
        @floop for agent in collect(allagents(model))
            update_vel!(agent, model; viscosity=viscosity)
        end
    end
    @timeit to "max reduction" begin
        max_vel = maximum(map(x->norm(x.vel), allagents(model)))
    end
    @timeit to "mean reduction" begin
        mean_vel = mean(map(x->norm(x.vel), allagents(model)))
    end
    # model.time_scale = max(0.1/max_vel, 1.)
    if max_vel*model.time_scale > getfield(model, :space).spacing || mean_vel*model.time_scale > 30
        model.time_scale /= 1.1
    end
    if model.time_scale < 0.9
        model.time_scale *= 1.01
    elseif model.time_scale > 1.1
        model.time_scale /= 1.01
    end
    @debug model.time_scale

    delta_time = time_ns() - ParticleLife.last_model_step_time
    ParticleLife.last_model_step_time = time_ns()
    fit!(ParticleLife.avg_model_step_duration[], delta_time/1e9)
    notify(ParticleLife.avg_model_step_duration)
end

agent_step!(agent, model) = move_agent!(agent, model, abmproperties(model)[:time_scale])
function update_vel!(agent::Particle, model::ABM; viscosity::Union{Nothing, Float64}=nothing)
    force = sum(
        let g = color_interact(agent.color, other.color, model),
            d = euclidean_distance(agent, other, model)
            (0 < d < 80 ? (g / d .* (agent.pos - other.pos)) : zero(SVector{2, Float64}))
        end
        for other in Agents.nearby_agents(agent, model, 80);
        init = zero(SVector{2, Float64}),
    )
    # push away from border
    force += 0.1*(max.(40 .- agent.pos, 0)
                 - max.(40 .- (spacesize(model) - agent.pos), 0))

    # combine past velocity and current force
    viscosity = (isnothing(viscosity) ? abmproperties(model)[:viscosity] : viscosity)
    agent.vel = agent.vel * (1-viscosity) + force
end

function run_sim(; to=TimerOutput())
    # this is basically it, but we want to rearrange the layout a bit.
    model = make_model(to)
    fig, ax, abmobs = with_theme(theme_dark()) do
        abmplot(model;
                ac=color_sym, as=8.0,  # agent color and size
                params=ParticleLife.properties,
                scatterkwargs=(; :markerspace=>:data),
                enable_inspection=false)
    end
    # fig[1,2] = content(fig[2,1])
    controls = content(fig[2,1][1,1])
    param_sliders = content(fig[2,1][1,2])
    update_button = content(param_sliders[2,1])

    # fig[1,1] = ax
    ax.width[] = 1000
    ax.height[] = 1500
    ui = fig[1,2] = GridLayout();
    param_sliders.width[] = 500
    ui[1,1] = param_sliders
    sg::SliderGrid = content(param_sliders[1,1])
    update_and_rand_button = param_sliders[2,1] = GridLayout();
    update_and_rand_button[1,1] = update_button
    rand_button = Button(update_and_rand_button[1,2], label="randomize", tellwidth=true)
    on(rand_button.clicks) do _
        for s in sg.sliders
            set_close_to!(s, rand(s.range[]))
        end
        update_button.clicks[] += 1
        abmproperties(model)[:time_scale] = 1.0
    end

    Label(ui[2,1], "----------------------")
    ui[3,1] = controls
    Label(ui[4,1], "----------------------")
    empty!(avg_model_step_duration.listeners)
    fps = throttle(0.5, @lift 1/value($avg_model_step_duration))
    fps_label = Label(ui[5,1], text="0.0 fps")
    on(fps) do fps
        fps_label.text = "$(round(fps, digits=2)) fps"
    end

    Makie.deleterow!(fig.layout, 2)
    fig
end

function make_video()
    with_theme(theme_dark()) do
        abmvideo("/tmp/foo.mp4", make_model(), ac=color_sym, as=8.0;
                scatterkwargs=(; :markerspace=>:data))
    end
end

properties=OrderedDict(
    :red_red       => -1:0.1:1,
    :red_green     => -1:0.1:1,
    :red_orange    => -1:0.1:1,
    :red_cyan      => -1:0.1:1,
    :green_red     => -1:0.1:1,
    :green_green   => -1:0.1:1,
    :green_orange  => -1:0.1:1,
    :green_cyan    => -1:0.1:1,
    :orange_red    => -1:0.1:1,
    :orange_green  => -1:0.1:1,
    :orange_orange => -1:0.1:1,
    :orange_cyan   => -1:0.1:1,
    :cyan_red      => -1:0.1:1,
    :cyan_green    => -1:0.1:1,
    :cyan_orange   => -1:0.1:1,
    :cyan_cyan     => -1:0.1:1,
    :viscosity     =>  0:.01:1,
)

end # module ParticleLife

#  Copyright 2017, Oscar Dowson
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################

#=
    This model can be used to build and run the model described in

        Dowson et al. (2017). A multi-stage stochastic optimization model of a
        dairy farm with spot-price uncertainty. Manuscript in preparation.

    The easiest way to run it is via the command line

        julia POWDER.jl "path/to/parameters.json"
=#
# @everywhere ENV["GRB_LICENSE_FILE"] = "C:/Users/odow003"
using SDDP, SDDPPro, JuMP, Gurobi, CPLEX, JSON

@everywhere function weeks(t::Int, T::Int=24)
    Tl = floor(Int, 52 / T)
    modulus = round(Int, T / (52 - Tl * T))
    offset = Tl * (t - 1) + floor(Int, (t-1) / modulus)
    if mod(t, modulus) == 0
        offset + (1:(Tl + 1))
    else
        offset + (1:Tl)
    end
end

"""
    buildPOWDER(parameters::Dict)

Construct the `SDDPModel` for POWDER given a dictionary of parameters.
"""
function buildPOWDER(parameters::Dict)


    # This function builds our value function
    function valuefunction(t, i)
        # form a discrete distribution
        noise_distribution = DiscreteDistribution(Float64.(parameters["model_two_observations"]))

        # the initial price in week 1
        initialprice = (parameters["initial_price"], 0.0)

        # minimum prices
        min_price = (0.0, 0.0)

        # maximum prices
        max_price = (12.0, 12.0)

        # our price model
        function modeldynamics(price, noise, month, markovstate)
            if month == 1
                return (price[1], price[2] + parameters["sales_curve"][1] * price[1])
            else
                gt = 0.96 * price[1] + 0.252 + noise
                at = price[2] + parameters["sales_curve"][month] * gt
                return (gt, at)
            end
        end
        return DynamicPriceInterpolation(
                 dynamics = modeldynamics,
            initial_price = initialprice,
                min_price = min_price,
                max_price = max_price,
                    noise = noise_distribution,
       lipschitz_constant = 5_000.0
        )
    end
    m = SDDPModel(
                        sense = :Max,
                       stages = 24,
                       # Change this to choose a different solver
                       # Method = 1 => Dual Simplex
                       solver = GurobiSolver(OutputFlag=0),
                     # solver = CplexSolver(CPX_PARAM_SCRIND=0, CPX_PARAM_LPMETHOD=2),
              objective_bound = 20_000, #parameters["objective_bound"],
              risk_measure = NestedAVaR(lambda=parameters["lambda"], beta=parameters["beta"]),
              value_function = valuefunction
                ) do sp, month



        # load the stagewise independent noises
        Ω = parameters["niwa_data"]
        Pₘ = parameters["maximum_pasture_cover"]  # maximum pasture-cover
        Pₙ = parameters["number_of_pasture_cuts"] # number of pasture growth curve cuts
        gₘ = parameters["maximum_growth_rate"]    # pasture growth curve coefficient
        β = parameters["harvesting_efficiency"]   # efficiency of harvesting pasture-cover into supplement
        ηₚ = parameters["pasture_energy_density"] # net energy content of pasture (MJ/kgDM)
        ηₛ = parameters["supplement_energy_density"] # net energy content of supplement (MJ/kgDM)

        # index of soil fertility estimated from average seasonal pasture growth
        κ = parameters["soil_fertility"] # actual growth was too low

        function futures_contribution(gt, w, t)
            y = 0.0
            for i in (t+1):length(w)
                y += w[i] * (0.96 ^ (i - t) * gt + 0.252 * sum(0.96^(j-1) for j in 1:(i-t)))
            end
            y
        end

        # pasture growth as a function of pasture cover
        g(p, gmax=gₘ, pmax=Pₘ) = 4 * gmax / pmax * p * (1 - p / pmax)
        # derivative of g(p) w.r.t. pasture cover
        dgdt(p, gmax=gₘ, pmax = Pₘ) = 4 * gmax / pmax * (1 - 2p / pmax)

        # Create states
        @states(sp, begin
            P >= 0, P₀ == parameters["initial_pasture_cover"] # pasture cover (kgDM/Ha)
            # Q >= 0, Q₀ == parameters["initial_storage"]       # supplement storage (kgDM)
            W >= 0, W₀ == parameters["initial_soil_moisture"] # soil moisture (mm)
            # C₀ are the cows milking during the stage
            C >= 0, C₀ == parameters["stocking_rate"]         # number of cows milking
            # need to bound this initially until we get some cuts
            -parameters["maximum_milk_production"] <= M <= parameters["maximum_milk_production"],      M₀ == 0.0                  # quantity of unsold milk
        end)

        # Create variables
        @variables(sp, begin
            b   >= 0 # quantity of supplement to feed (kgDM)
            h   >= 0 # quantity of supplement to harvest (kgDM/Ha)
            i   >= 0 # irrigate farm (mm/Ha)
            fₛ  >= 0 # feed herd stored pasture (kgDM)
            fₚ  >= 0 # feed herd pasture (kgDM)
            ev  >= 0 # evapotranspiration rate (mm/day)
            gr  >= 0 # potential growth (kgDM/Ha/Day)
            mlk >= 0 # milk production (MJ)
            milk_sales >= 0
            # milk_buys  >= 0
            #=
                Dummy variables for later reporting
            =#
            cx   # the stage objective  excl. penalties
            milk # kgMS
            #=
                Penalties
            =#
            Δ[i=1:2] >= 0
        end)

        # Build an expression for the energy required to save space later
        @expressions(sp, begin
            energy_req, sum(
                parameters["stocking_rate"] * (
                    parameters["energy_for_pregnancy"][week] +
                    parameters["energy_for_maintenance"] +
                    parameters["energy_for_bcs_dry"][week]
                    ) +
                    C₀ * ( parameters["energy_for_bcs_milking"][week] -
                            parameters["energy_for_bcs_dry"][week])
                            for week in weeks(month)
            )
        end)

        average_fertility = mean(κ[week]/7 for week in weeks(month))
        average_energy_content_of_milk = mean(parameters["energy_content_of_milk"][week] for week in weeks(month))
        @constraints(sp, begin
            # State transitions
            P <= P₀ + 7*length(weeks(month)) * gr - h - fₚ
            P <= Pₘ
            # Q <= Q₀ + β*h - fₛ
            fₛ == 0.0

            C <= C₀ # implicitly C == C₀ - u | u ≥ 0
            W <= parameters["maximum_soil_moisture"]

            milk <= mlk / average_energy_content_of_milk

            M <= M₀ + milk - milk_sales# + milk_buys
            # energy balance
            ηₚ * (fₚ + fₛ) + ηₛ * b >= energy_req + mlk

            # maximum milk
            mlk <= sum(parameters["max_milk_energy"][week] for week in weeks(month)) * C₀
            # minimum milk
            milk >= sum(parameters["min_milk_production"] for week in weeks(month)) * C₀

            # pasture growth constraints
            gr <= average_fertility * ev
            # +1e-5 is used to avoid gr being slightly negative (although should
            # be fixed in SDDP.jl - see #72
            [pbar=linspace(0,Pₘ, Pₙ)], gr <= g(pbar) + dgdt(pbar) * ( P₀ - pbar) + 1e-2

            # max irrigation
            i <= parameters["maximum_irrigation"]
            milk_sales <= parameters["max_milk_contracting"]
            h <= 0
        end)

        # average the noise values
        noise = [ Dict{String, Float64}(
                "evapotranspiration" => mean(
                    ω["evapotranspiration"] for week in weeks(month) for ω in Ω[week] if ω["year"] == year
                ),
                "rainfall" => mean(
                    ω["rainfall"] for week in weeks(month) for ω in Ω[week] if ω["year"] == year
                )
            ) for year in 11:20 ]

        @rhsnoises(sp, ω = noise, begin
            # evapotranspiration limited by potential evapotranspiration
            ev <= ω["evapotranspiration"]
            # less than accounts for drainage
            W <= W₀ - ev + ω["rainfall"] + i
        end)

        if weeks(month)[end] + 1 >= parameters["maximum_lactation"]
            # dry off by end of week 44 (end of may)
            @constraint(sp, C <= 0)
        end

        # a maximum rate of supplementation - for example due to FEI limits
        cow_per_day = parameters["stocking_rate"] * 7 * length(weeks(month))
        @constraints(sp, begin
            Δ[1] >= 0
            Δ[1] >= cow_per_day * (0.00 + 0.25 * (b / cow_per_day - 3))
            Δ[1] >= cow_per_day * (0.25 + 0.50 * (b / cow_per_day - 4))
            Δ[1] >= cow_per_day * (0.75 + 1.00 * (b / cow_per_day - 5))
        end)

        if month != 24
            # auction = findlast(x->x<=stage, parameters["auction_weeks"])
            @stageobjective(sp, (price) -> (
                (
                    price[2] + futures_contribution(price[1], parameters["sales_curve"], month)-
                    parameters["transaction_cost"]
                ) * milk_sales -
                # cost of supplement ($/kgDM). Incl %DM from wet, storage loss, wastage
                parameters["supplement_price"] * b -
                parameters["cost_irrigation"] * i -   # cost of irrigation ($/mm)
                parameters["harvest_cost"] * h -      # cost of harvest    ($/kgDM)
                Δ[1] - # penalty from excess feeding due to FEI limits
                1e2 * Δ[2] + # penalise low pasture cover
                0.0001*W # encourage soil moisture to max
            ))
        else
            @stageobjective(sp, (price) -> (
                price[2] * M - # forced to sell at Fonterra price
                # cost of supplement ($/kgDM). Incl %DM from wet, storage loss, wastage
                parameters["supplement_price"] * b -
                parameters["cost_irrigation"] * i -   # cost of irrigation ($/mm)
                parameters["harvest_cost"] * h -      # cost of harvest    ($/kgDM)
                Δ[1] - # penalty from excess feeding due to FEI limits
                1e2 * Δ[2] + # penalise low pasture cover
                0.0001*W # encourage soil moisture to max
            ))

            # end with minimum pasture cover
            @constraints(sp, begin
                P + Δ[2] >= parameters["final_pasture_cover"]
            end)
        end
    end
    return m
end

# """
#     simulatemodel(fileprefix::String, m::SDDPModel, NSIM::Int, prices, Ω)""
#
# This function simulates the SDDPModel `m` `NSIM` times and saves:
#  - the HTML visualization of the simulation at `fileprefix.html`
#  - the results dictionary as a Julia serialized file at `fileprefix.results`
# """
function simulatemodel(fileprefix::String, m::SDDPModel, parameters)
    NSIM = parameters["number_simulations"]
    # fileprefix = replace(fileprefix, " ", "")
    srand(parameters["random_seed"])
    results = simulate(m, NSIM, [:P, :C, :C₀, :W, :M, :fₛ, :fₚ, :gr, :mlk, :ev, :b, :i, :h, :Δ, :milk, :price, :milk_sales])

    p = SDDP.newplot()
    SDDP.addplot!(p, 1:NSIM, 1:24, (i,t)->results[i][:stageobjective][t], title="Objective", cumulative=true)
    SDDP.addplot!(p, 1:NSIM, 1:24, (i,t)->results[i][:P][t], title="Pasture Cover")
    # SDDP.addplot!(p, 1:NSIM, 1:12, (i,t)->results[i][:Q][t], title="Supplement")
    SDDP.addplot!(p, 1:NSIM, 1:24, (i,t)->results[i][:W][t], title="Soil Moisture")
    SDDP.addplot!(p, 1:NSIM, 1:24, (i,t)->results[i][:C][t], title="Cows Milking")
    SDDP.addplot!(p, 1:NSIM, 1:24, (i,t)->results[i][:M][t], title="Unsold Milk")
    SDDP.addplot!(p, 1:NSIM, 1:24, (i,t)->results[i][:milk][t] / 7 / 3 / length(weeks(t)), title="Milk Production / Cow")
    SDDP.addplot!(p, 1:NSIM, 1:24, (i,t)->results[i][:gr][t], title="Growth")
    SDDP.addplot!(p, 1:NSIM, 1:24, (i,t)->results[i][:ev][t], title="Actual Evapotranspiration")
    # SDDP.addplot!(p, 1:NSIM, 1:12, (i,t)->parameters["niwa_data"][t][results[i][:noise][t]]["evapotranspiration"], title="Potential Evapotranspiration")
    # SDDP.addplot!(p, 1:NSIM, 1:12, (i,t)->parameters["niwa_data"][t][results[i][:noise][t]]["rainfall"], title="Rainfall")
    SDDP.addplot!(p, 1:NSIM, 1:24, (i,t)->results[i][:price][t][2], title="Price")
    SDDP.addplot!(p, 1:NSIM, 1:24, (i,t)->results[i][:price][t][1], title="GDT")
    SDDP.addplot!(p, 1:NSIM, 1:24, (i,t)->results[i][:milk_sales][t], title="Milk Sales")
    SDDP.addplot!(p, 1:NSIM, 1:24, (i,t)->results[i][:mlk][t], title="Milk")
    SDDP.addplot!(p, 1:NSIM, 1:24, (i,t)->results[i][:fₚ][t], title="Feed Pasture")
    SDDP.addplot!(p, 1:NSIM, 1:24, (i,t)->results[i][:fₛ][t], title="Feed Silage")
    SDDP.addplot!(p, 1:NSIM, 1:24, (i,t)->results[i][:b][t], title="Feed PKE")
    SDDP.addplot!(p, 1:NSIM, 1:24, (i,t)->results[i][:i][t], title="Irrigation")
    SDDP.addplot!(p, 1:NSIM, 1:24, (i,t)->results[i][:h][t], title="Harvest")
    SDDP.show("$(fileprefix).html", p)
    SDDP.save!("$(fileprefix).results", results)
    results
end

"""
    runPOWDER(parameterfile::String)

Run POWDER model and produce some nice pictures and statistics
"""
function runPOWDER(parameterfile::String)

    parameters = JSON.parsefile(parameterfile) # parse the parameters
    srand(parameters["random_seed"])
    m = buildPOWDER(parameters)                # build the sddp model
    name = parameters["model_name"]            # name to save files

    # solve the model
    solve(m,
        max_iterations=5000,
        cut_output_file="$(name).cuts",
        log_file="$(name).log",
        # cut_selection_frequency = 100,
        print_level=2
    )

    # save a serialized version so we can return to it later
    SDDP.savemodel!("$(name).model", m)

    # simulate it
    results = simulatemodel(name, m, parameters)

    # build summary results table
    headers = [
        "Lactation Length (Weeks) ",
        "Milk Production (kg)",
        "per Hectare",
        "per Cow",
        "Milk Revenue (\\\$/Ha)",
        "Feed Consumed (t/Ha)",
        "Pasture",
        "Palm Kernel",
        "\\% Feed Imported",
        "Palm Kernel Expense (\\\$/Ha)",
        "Fixed Expense (\\\$/Ha)",
        "Operating Profit (\\\$/Ha)",
        "Fat Evaluation Index Penalty"
    ]

    benchmark = [
        38.6,
        "",
        1193,
        398,
        7158,
        "",
        12.15,
        2.85,
        19,
        1425,
        3536,
        2197,
        "-"
    ]
    function milkrevenue(sim)
        y = 0.0
        for t in 1:11
            y += sim[:milk_sales][t] * (sim[:price][t][1] - parameters["transaction_cost"])
        end
        return y + sim[:price][end][2] * sim[:M][end]
    end
    simquant(x) = quantile(x, [0.0, 0.25, 0.5, 0.75, 1.0])
    data = Array{Any}(13, 5)
    data[1,:] = simquant([sum(sim[:C₀]) / parameters["stocking_rate"] for sim in results])
    data[2,:] = ""
    data[3,:] = simquant([sum(sim[:milk]) for sim in results])
    data[4,:] = data[3,:] / parameters["stocking_rate"]
    data[5,:] = simquant(milkrevenue.(results))
    data[6,:] = ""
    data[7,:] = simquant([sum(sim[:fₚ]) + sum(sim[:fₛ]) for sim in results]) / 1000
    data[8,:] = simquant([sum(sim[:b]) for sim in results]) / 1000
    data[9,:] = 100*simquant([sum(sim[:b]) ./ (sum(sim[:b]) + sum(sim[:fₛ]) + sum(sim[:fₚ])) for sim in results])
    data[10,:] = data[8,:] * 1000 * parameters["supplement_price"]
    data[11,:] = parameters["fixed_cost"]
    data[12,:] = simquant([sum(sim[:stageobjective]) + sum(sim[:Δ][1]) for sim in results]) - parameters["fixed_cost"]
    data[13,:] = simquant([sum(sim[:Δ][1]) for sim in results])

    # print a formatted table for copying into latex
    roundings = [1, 0, 0, 0, 0, 0, 2, 2, 1, 0, 0, 0, 0, 0]
    hlines = [true, false, false, false, true, false, false, false, false, true, true, false, true, false]
    open("$(name).data.tex", "w") do io
        for i in 1:size(data, 1)
            print(io, headers[i])
            for j in 1:size(data, 2)
                print(io, " & ")
                if data[i,j] != ""
                    if roundings[i] == 0
                        print(io, round(Int, data[i, j]))
                    else
                        print(io, round(data[i, j], roundings[i]))
                    end
                end
            end
            println(io, " & ", benchmark[i], "\\\\")
            if hlines[i]
                println(io, "\\hline")
            end
        end
    end
end

# julia POWDER.jl "path/to/parameters.json"
if length(ARGS) > 0
    runPOWDER(ARGS[1])
end

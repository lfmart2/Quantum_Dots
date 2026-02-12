using Plots
using LaTeXStrings
using LinearAlgebra
using Statistics
using Plots.PlotMeasures
using Optim
using Optimization
using OptimizationBBO
using CSV,  DataFrames
using MAT

############################################################################################################
######################### Function with the 1P and 1S transition energies #################################
############################################################################################################
# Function to calculate the q-parameter for 1P transition
function qparam_1P(η,W,t_1S,t_1P)
    q_max = (1.0 + η*(t_1S[1])^2) / (t_1P[1]*t_1S[1]/W + 1.0) 
    return (1.0 .+ η .* t_1S.^2)./(q_max*(t_1S .* t_1P ./ W .+ 1.0))
end

# Function to calculate the sum of squared errors for 1P transition
function sse_1P(params, t_1S, t_1P, q)
    η, W = params
    sum( abs.(q - qparam_1P(η,W,t_1S,t_1P) ).^2 )
end

function optimize_params_constrained_1P(t_1S, t_1P, q)
    lower = [0.0, 0.0]
    upper = [30.0, 30.0]  # Updated upper bounds for η and W
    initial_params = [10.0, 10.0]

    # Define the objective function with bounds
    function obj_with_bounds(params)
        η, W = params
        if η >= upper[1] || W >= upper[2] || η <= lower[1] || W <= lower[2]
            return Inf  # Penalize points outside the feasible region
        else
            return sse_1P(params, t_1S, t_1P, q)
        end
    end

    res = optimize(obj_with_bounds, lower, upper, initial_params, NelderMead())
    return Optim.minimizer(res)
end



function qq_ds_Fano_1P()
    D₀  = collect(range(0.64, 1.8, length=1000)); # Quantum dot sizes in meters
    D00 = [0.64, 0.77, 1.75]
    ## Paramateros to fit
    # tt_1S  = t_dis_1S(D00)./D00.^2
    # tt_1P  = t_dis_1P(D00)./D00.^2
    tt_1S  = t_dis_1S(D00)
    tt_1P  = t_dis_1P(D00)
    tt_1P  = tt_1P./maximum(tt_1S)
    tt_1S  = tt_1S./maximum(tt_1S)
    
    # S_tt   = t_dis_1S(D₀)./D₀.^2
    # P_tt   = t_dis_1P(D₀)./D₀.^2
    S_tt   = t_dis_1S(D₀)
    P_tt   = t_dis_1P(D₀)
    P_tt = P_tt./maximum(S_tt)
    S_tt = S_tt./maximum(S_tt)
    

    q  = [13.9, 13.1, 10.7] #[0.54, 0.26, 0.21]  #
    q_nom = q./maximum(q)
    

    # η = 6.0; W = 5;

    ## Experimental data
    q_err = [0.45/25.74, 0.64/50.38, 0.50/50.95]./maximum(q) # [14.2, 0.45, 0.64, 0.50]
    QD    = ["ZnS₀.₆₄", "ZnS₀.₇₇", "ZnS₁.₇₅"] #["Core", "ZnS₀.₆₄", "ZnS₀.₇₇", "ZnS₁.₇₅"]



    # # parameters for 1/R^6
    # S_tt = t_dis_1S(D₀)./D₀.^6.0./maximum(t_dis_1S(D₀)./D₀.^6.0)
    # P_tt = t_dis_1P(D₀)./D₀.^6.0./maximum(t_dis_1S(D₀)./D₀.^6.0)
    # tt_1S = t_dis_1S(D00)./D00.^6.0./maximum(t_dis_1S(D00)./D00.^6.0)
    # tt_1P = t_dis_1P(D00)./D00.^6.0./maximum(t_dis_1S(D00)./D00.^6.0)
    # D00 = D00.^6; D00 = 1.0./D00; D₀ = D₀.^6; D₀ = 1.0./D₀


    println(optimize_params_constrained_1P(tt_1S,tt_1P,q_nom))
    η, W = optimize_params_constrained_1P(tt_1S,tt_1P,q_nom)
    # W = 0.04
    ## Plot the q-parameter via eq. 2.118: Γ + q for small Γ
    p1 = plot()
    plot!(p1,D₀, qparam_1P(η,W,S_tt,P_tt),
        label=false,
        lw=2,
        ylabel = L"\bar{q}'/\bar{q}'_{max}",
        xlabel = "Diameter t(r)")
    for i in eachindex(tt_1S)
        scatter!(p1, (D00[i], q_nom[i]),
            mc=i,
            yerr = [q_err[i]],
            markershape=:square,
            ma=1,
            ms = 5  ,
            label="$(QD[i])-Experimental")
        scatter!(p1, (D00[i], qparam_1P(η,W,tt_1S,tt_1P)[i]),
            mc=i,
            ms=5,
            ma=1,
            label="Theoretical")
        plot!(p1,legendfontsize = 11, legendcolumns=2,
            xguidefontsize=16, yguidefontsize=16,
            xtickfontsize=11, ytickfontsize=11,
            grid = false, legend = :topleft)
            # plot!(p5, legend = false)
    end
    display(p1)
    return D₀, qparam_1P(η,W,S_tt,P_tt)
end

function qq_sz_Fano_1P()
    D00 = [2.6, 3.0, 3.8, 5.6] # Quantum dot sizes in nanometers
    D₀ = collect(range(2.6, 7, length=1000)); # Quantum dot sizes in meters
    # tt_1S  = t_size_1S(D00)./maximum(t_size_1S(D00))
    # tt_1P  = t_size_1P(D00)./maximum(t_size_1S(D00))
    # S_tt = t_size_1S(D₀)./maximum(t_size_1S(D₀))
    # P_tt = t_size_1P(D₀)./maximum(t_size_1S(D₀))
    ## Paramateros to fit
    # q   = [1.37, 1.32, 0.79, 0.159] # [29.8, 22.9, 20.2, 4.41]
    q   = [29.8, 22.9, 20.2, 4.41]
    q_nom = q./maximum(q)
    # q_err = [7.3/21.75, 3.6/17.35, 6.8/25.57, 1.1/27.74]./maximum(q)
    q_err = [7.3, 3.6, 6.8, 1.1]./maximum(q)
    QD  = ["CdSe 525", "CdSe 550", "CdSe 580", "CdSe 620"]


    # # parameters for 1/R^6
    # tt_1P = t_size_1P(D00)./D00.^6.0./maximum(t_size_1S(D00)./D00.^6.0)
    # tt_1S = t_size_1S(D00)./D00.^6.0./maximum(t_size_1S(D00)./D00.^6.0)
    # S_tt = t_size_1S(D₀)./D₀.^6.0./maximum(t_size_1S(D₀)./D₀.^6.0)
    # P_tt = t_size_1P(D₀)./D₀.^6.0./maximum(t_size_1S(D₀)./D₀.^6.0)
    # D00 = D00.^6; D00 = 1.0./D00; D₀ = D₀.^6; D₀ = 1.0./D₀
    tt_1S  = t_size_1S(D00)./D00.^2
    tt_1P  = t_size_1P_2(D00)./D00.^2
    tt_1P  = tt_1P./maximum(tt_1S)
    tt_1S  = tt_1S./maximum(tt_1S)
    
    S_tt   = t_size_1S(D₀)./D₀.^2
    P_tt   = t_size_1P_2(D₀)./D₀.^2
    
    P_tt   = P_tt./maximum(S_tt)
    S_tt   = S_tt./maximum(S_tt)


    println(optimize_params_constrained_1P(tt_1S,tt_1P,q_nom))
    η, W = optimize_params_constrained_1P(tt_1S,tt_1P,q_nom)
    # η = 0.46; W = 29

    p1 = plot()
    p2 = plot()

    
    # plot!(p1,S_tt, qparam(η,W,S_tt,P_tt),
    #     label=false,
    #     lw =2,
    #     ylabel = L"\bar{q}'/\bar{q}'_{max}",
    #     xlabel = "normalized electronic coupling")

    
    plot!(p2,vec(D₀),qparam_1P(η,W,S_tt,P_tt),
            label=false,
            lw =2,
            ylabel = L"\bar{q}'/\bar{q}'_{max}",
            xlabel = "Diameter (nm)")

    # Create an example DataFrame
    df = DataFrame(d0_size = D₀, q_Fano = qparam_1P(η,W,S_tt,P_tt))

    # Write the DataFrame to a CSV file
    CSV.write("q_size.csv", df)
    for i in 1:length(tt_1S)
        # scatter!(p1, (tt_1S[i], q_nom[i]),
        #     mc=i,
        #     yerr = [q_err[i]],
        #     markershape=:square,
        #     ma=1,
        #     label=QD[i]*" - Exp")
        # scatter!(p1, (tt_1S[i], qparam(η,W,tt_1S,tt_1P)[i]),
        #     mc=i,
        #     ms=4,
        #     ma=1,
        #     label="Theoretical")
        # plot!(p1,
        #     legend = :topright,
        #     legendcolumns=2,
        #     legendfontsize = 7,
        #     xguidefontsize=15, yguidefontsize=15,
        #     xtickfontsize=10, ytickfontsize=10,
        # )

        scatter!(p2, (D00[i], q_nom[i]),
            mc=i,
            yerr = [q_err[i]],
            markershape=:square,
            ma=1,
            label=QD[i]*"-Experimental")
        scatter!(p2, (D00[i], qparam_1P(η,W,tt_1S,tt_1P)[i]),
            mc=i,
            ms=5,
            ma=1,
            label="Theoretical")
        plot!(p2,legendfontsize = 11, legendcolumns = 2,
            xguidefontsize=16, yguidefontsize=16,
            xtickfontsize=11, ytickfontsize=11, grid = false,
            ylims = (0.0, 1.5))
        # plot!(p5, legend = false)
    end
    return p2
    # return D₀, qparam_1P(η,W,S_tt,P_tt)
    
end

# matwrite("ds_dep_1P.mat", Dict(
#                "QD_ds" => tmp1,
#                "q_ds" => tmp2
#        ))

function t_coupling_plots()

    D₀_dis  = collect(range(0.64, 1.8, length=1000))
    t_1S_dis   = t_dis_1S(D₀_dis)
    t_1P_dis   = t_dis_1P(D₀_dis)
    tt_1P_dis   = t_1P_dis./maximum(t_1S_dis)
    tt_1S_dis   = t_1S_dis./maximum(t_1S_dis)
    


    D₀_size = collect(range(2.6, 7, length=1000));
    t_1S_size  = t_size_1S(D₀_size)
    t_1P_size  = t_size_1P_2(D₀_size)
    tt_1S_size  = t_1S_size./maximum(t_1S_size)
    tt_1P_size  = t_1P_size./maximum(t_1S_size)
    

    # η = 6.0; W = 5;

    ## Experimental data
    QD    = ["ZnS₀.₆₄", "ZnS₀.₇₇", "ZnS₁.₇₅"] #["Core", "ZnS₀.₆₄", "ZnS₀.₇₇", "ZnS₁.₇₅"]



    ## Plot the q-parameter via eq. 2.118: Γ + q for small Γ
    p1 = plot()
    plot!(p1,D₀_dis, tt_1S_dis,
        label="1S t coupling - distance dependent",
        lw=2,
        ylabel = L"\bar{q}'/\bar{q}'_{max}",
        xlabel = "Diameter t(r)")
    
    plot!(p1,D₀_size, tt_1S_size,
        label="1S t coupling - size dependent",
        lw=2)
    # Create an example DataFrame
    df = DataFrame(d0_dis = D₀_dis, t1S_dis = tt_1S_dis, t1P_dis = tt_1P_dis)

    # Write the DataFrame to a CSV file
    CSV.write("distance_dep.csv", df)
    p2 = plot()
    plot!(p2,D₀_dis, tt_1P_dis,
        label="1P t coupling - distance dependent",
        lw=2)
    plot!(p2,D₀_size, tt_1P_size,
        label="1P t coupling - size dependent",
        lw=2)

    display(p1)
    display(p2)

    # Create an example DataFrame
    df = DataFrame(d0_size = D₀_size, t1S_size = tt_1S_size, t1P_size = tt_1P_size)

    # Write the DataFrame to a CSV file
    CSV.write("size_dep.csv", df)
    return p1, p2
end

function qq_ds_Fano_1P_sensitivity()
    D₀  = collect(range(0.64, 1.8, length=1000)); # Quantum dot sizes in meters
    D00 = [0.64, 0.77, 1.75]
    ## Paramateros to fit
    tt_1S  = t_dis_1S(D00)
    tt_1P  = t_dis_1P(D00)
    tt_1P  = tt_1P./maximum(tt_1S)
    tt_1S  = tt_1S./maximum(tt_1S)
    
    S_tt   = t_dis_1S(D₀)
    P_tt   = t_dis_1P(D₀)
    P_tt = P_tt./maximum(S_tt)
    S_tt = S_tt./maximum(S_tt)
    

    q  = [13.9, 13.1, 10.7] #[0.54, 0.26, 0.21]  #
    q_nom = q./maximum(q)



    println(optimize_params_constrained_1P(tt_1S,tt_1P,q_nom) )
    η, W = optimize_params_constrained_1P(tt_1S,tt_1P,q_nom) 

    # Create an example DataFrame
    df = DataFrame(D0 = D₀, q_Fano = qparam_1P(η,W,S_tt,P_tt))

    # Write the DataFrame to a CSV file
    CSV.write("distance_dep_sensitivity.csv", df)
end

function qq_sz_Fano_1P_sensitivity()
    D00 = [2.6, 3.0, 3.8, 5.6] # Quantum dot sizes in nanometers
    D₀ = collect(range(2.6, 7, length=1000)); # Quantum dot sizes in meters
    tt_1S  = t_size_1S(D00)./maximum(t_size_1S(D00))
    tt_1P  = t_size_1P_2(D00)./maximum(t_size_1S(D00))
    S_tt = t_size_1S(D₀)./maximum(t_size_1S(D₀))
    P_tt = t_size_1P_2(D₀)./maximum(t_size_1S(D₀))
    ## Paramateros to fit
    q   = [1.37, 1.32, 0.79, 0.159] # [29.8, 22.9, 20.2, 4.41]
    q_nom = q./maximum(q)
    




    println(optimize_params_constrained_1P(tt_1S,tt_1P,q_nom))
    η, W = optimize_params_constrained_1P(tt_1S,tt_1P,q_nom)


    # Create an example DataFrame
    df = DataFrame(D0 = D₀, q_Fano = qparam_1P(η,W,S_tt,P_tt))

    # Write the DataFrame to a CSV file
    CSV.write("size_dep_sensitivity.csv", df)
end




Plots.default(titlefont = ("times"), legendfont = ("times"), xlabelfont="Helvetica")


## Define the q-parameter function. The coupling parameters should be normalized
function qparam(η,W,t_1S)
    q_max = (1.0 + η / (1/W + 1.0) )
    return (1.0 .+ η .* t_1S.^2)./(q_max*(t_1S.^2 ./ W .+ 1.0))
end
# tt_dis_1S  = t_dis_1S()./maximum(t_dis_1S())
# tt_dis_1P  = t_dis_1P()./maximum(t_dis_1S())
# tt_size_1S = t_size_1S()./maximum(t_size_1S())
# tt_size_1P = t_size_1P()./maximum(t_size_1S())
# q_dis  = [78.7, 13.9, 13.1, 10.7]
# q_size   = [29.8, 22.9, 20.2, 4.41]
# q_s_nom = q_size./maximum(q_size)
# q_d_nom = q_dis./maximum(q_dis)

function sse_1S(params, t_1S, q)
    η, W = params
    n = length(t_1S)
    sum( abs.(q - qparam(η,W,t_1S) ).^2 )
end

# x = range(2, 8, length = 100)
# y = range(1.0, 20.0, length = 100)
# f_d(x, y) = sum( abs(q_d_nom[i] - qparam(x,y,tt_dis[i],maximum(tt_dis)) ) for i in 1:length(tt_dis))
# f_s(x, y) = sum( abs(q_s_nom[i] - qparam(x,y,tt_size[i],maximum(tt_size)) ) for i in 1:length(tt_size))

function optimize_params_constrained_1S(t_1S, q)
    lower = [0.0, 0.0]
    upper = [20.0, 20.0]  # Updated upper bounds for η and W
    initial_params = [1.0, 7.0]

    # Define the objective function with bounds
    function obj_with_bounds(params)
        η, W = params
        if η >= upper[1] || W >= upper[2] || η <= lower[1] || W <= lower[2]
            return Inf  # Penalize points outside the feasible region
        else
            return sse_1S(params, t_1S, q)
        end
    end

    res = optimize(obj_with_bounds, lower, upper, initial_params, NelderMead())
    return Optim.minimizer(res)
end

function qq_ds_Fano_1S()
    D₀  = collect(range(0.60, 1.8, length=1000)); # Quantum dot sizes in meters
    D00 = [0.64, 0.77, 1.75]
    ## Paramateros to fit
    tt_1S  = t_dis_1S(D00)./maximum(t_dis_1S(D00))
    S_tt = t_dis_1S(D₀)./maximum(t_dis_1S(D00))

    q  = [0.54, 0.26, 0.21]  # # [78.7, 13.9, 13.1, 10.7]
    q_nom = q./maximum(q)
    println(optimize_params_constrained_1S(tt_1S,q_nom))
    η, W = optimize_params_constrained_1S(tt_1S,q_nom)
    # η = 6.0; W = 5;

    ## Experimental data
    q_err = [0.45/25.74, 0.64/50.38, 0.50/50.95]./maximum(q)# [14.2, 0.45, 0.64, 0.50]./maximum(q)
    QD    = ["ZnS₀.₆₄", "ZnS₀.₇₇", "ZnS₁.₇₅"] # ["Core", "ZnS₀.₆₄", "ZnS₀.₇₇", "ZnS₁.₇₅"]
    t = range(0.02,1.0, length=1000);

    ## Plot the q-parameter via eq. 2.118: Γ + q for small Γ
    p1 = plot()
    plot!(p1,D₀, qparam.(η,W,S_tt),
        label=false,
        lw=2,
        ylabel = L"\bar{q}'/\bar{q}'_{max}",
        xlabel = "Shell thickness (nm)",
        xlabelfont="Helvetica"
        )
    for i in eachindex(tt_1S)
        scatter!(p1, (D00[i], q_nom[i]),
            mc=i,
            yerr = [q_err[i]],
            markershape=:square,
            ma=1,
            ms = 5  ,
            label="$(QD[i])-Experimental")
        scatter!(p1, (D00[i], qparam(η,W,tt_1S[i])),
            mc=i,
            ms=5,
            ma=1,
            label="Theoretical")
        plot!(p1,legendfontsize = 11, legendcolumns=2,
            xguidefontsize=16, yguidefontsize=16,
            xtickfontsize=11, ytickfontsize=11,
            grid = false, legend = :topright)
            # plot!(p5, legend = false)
    end

    # plot!(p5, ylims = (0,y_lim))
    return p1 # D₀, qparam(η,W,S_tt)
end

function qq_sz_Fano_1S()
    D00 = [2.6, 3.0, 3.8, 5.6] # Quantum dot sizes in nanometers
    D₀ = collect(range(2.6, 7, length=1000)); # Quantum dot sizes in meters
    
    tt_1S  = t_size_1S(D00)./maximum(t_size_1S(D00))
    S_tt = t_size_1S(D₀)./maximum(t_size_1S(D00))
    ## Paramateros to fit
    q   = [29.8, 22.9, 20.2, 4.41]
    q_nom = q./maximum(q)
    q_err = [7.3, 3.6, 6.8, 1.1]./maximum(q)
    QD  = ["CdSe 525", "CdSe 550", "CdSe 580", "CdSe 620"]

    println(optimize_params_constrained_1S(tt_1S,q_nom))
    η, W = optimize_params_constrained_1S(tt_1S,q_nom)
    p1 = plot()
    p2 = plot()
    
    # plot!(p1,S_tt, qparam(η,W,S_tt,P_tt),
    #     label=false,
    #     lw =2,
    #     ylabel = L"\bar{q}'/\bar{q}'_{max}",
    #     xlabel = "normalized electronic coupling")

    
    plot!(p2,D₀,qparam(η,W,S_tt),
            label=false,
            lw =2,
            ylabel = L"\bar{q}'/\bar{q}'_{max}",
            xlabel = "Diameter (nm)")

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
        scatter!(p2, (D00[i], qparam(η,W,tt_1S)[i]),
            mc=i,
            ms=5,
            ma=1,
            label="Theoretical")
        plot!(p2,legendfontsize = 11, legendcolumns = 2,
            xguidefontsize=16, yguidefontsize=16,
            xtickfontsize=11, ytickfontsize=11, grid = false)
        # plot!(p5, legend = false)
    end
    return p2
    
end

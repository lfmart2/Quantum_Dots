using LaTeXStrings
using LinearAlgebra
using Optimization
using OptimizationBBO
using NumericalIntegration
using Plots
using SpecialFunctions
using JLD2, CSV, DataFrames


########################################################################
########## Codes for the electronic coupling calculation ###############
##########          for the size dependance              ###############
########################################################################

# Definition of the conduction band edge for the CdSe quantum dot:
# Input parameter: dis - in nanometers
function cond_bnd_edge(dis::Float64)
    return -3.49 + 2.97*dis^(-1.24)
end

# Electronic coupling calculation. This code is used to calculate the electronic coupling energy at varius QD sizes.
# This code does not include the correction for the 1P electrons
function t_size_1S(D00::Vector{Float64})
    # Constants
    ħ = 1.054e-34;  # Planck's constant
    m = 9.10938356e-31;  # Electron mass

    # Quantum dot size
    R₀ = D00 .* 1e-9 ./2 # [2.6e-9, 3.0e-9, 3.8e-9, 5.6e-9]./2; # Quantum dot sizes in meters
    # R00 = [2.6, 3.0, 3.8, 5.6] # Quantum dot sizes in nanometers
    # Effective mass of electron
    mₑ = 0.13 * m;

    # Electron charge:
    q_e = 1.602176565e-19

    # Define initial potential energy
    V0 = cond_bnd_edge.(D00).*q_e ;

    # Bound state criterion
    r0 = -2*mₑ*V0.*R₀.^2/ħ^2

    # Start a new local scope
    x_a = zeros(length(V0)); y_a = zeros(length(V0))
    x = range(0.001,15, length=1000)
    y = range(0.001,15, length=1000)
    for i in 1:length(V0)
        xx = range(2,3, length=1000)
        x_a[i] = xx[convert.(Int64,argmin(abs.(map(x1->x1^2 + x1^2*(cot(x1))^2-r0[i],xx))))]
        y_a[i] = -x_a[i]*cot(x_a[i])
    end 

    q = y_a./R₀;
    k₀ = x_a./R₀;

    t = zeros(length(V0))    
    for i in 1:length(V0)
        r1 = range(0.00000000001,R₀[i], length=1000);
        r2 = range(R₀[i],8.0*10^-9, length=1000);

        R₁ = sin.(k₀[i] * r1) ./ r1;
        R₂ = exp.(-q[i] * r2) ./ r2;

        a = integrate(r1, abs.(R₁.*r1).^2)
        b = integrate(r2, abs.(R₂.*r2).^2)

        A_cnst = exp(-2*q[i]*R₀[i]) / (sin(k₀[i] * R₀[i])^2)
        
        B₀ = 1.0 / sqrt(a*A_cnst + b)
        t[i] = B₀*integrate(r2, R₂.*(r2).^2) # B₀*R₂[1]*R₀[i]^2
    end 

    return t
end

# Electronic coupling calculation. This code is used to calculate the electronic coupling energy at varius QD sizes.
# This code include the correction for the 1P electrons
function t_size_1P(D00::Vector{Float64})
    # Constants
    ħ = 1.054e-34;  # Planck's constant
    m = 9.10938356e-31;  # Electron mass

    # Quantum dot size
    R₀ = D00 .* 1e-9 ./2 # # Quantum dot sizes (diameter) in nanometers
    # Effective mass of electron
    mₑ = 0.13 * m;

    # Electron charge:
    q_e = 1.602176565e-19

    # Define initial potential energy
    V0 = cond_bnd_edge.(D00).*q_e ;

    # Bound state criterion
    r0 =  -2*mₑ*V0 .* R₀ .^2 ./ ħ^2

    # Definition of the spherical Bessel and Hankel functions
    sphericalhankelj(ν, x) = √(π/2x).*hankelh1(ν+1/2, x)
    sphericalbesselj(ν, x) = √(π/2x).*besselj(ν+1/2, x)


    E_q = zeros(length(V0)); #Quantized values for the Energy
    # Set titles for the plots
    t = zeros(length(V0))
    for i in eachindex(V0)
        # Start a new local scope
        y(x) = sqrt.(r0[i] .- x.^2)
        
        f_in(x) = x .* sphericalbesselj.(0,x)./ sphericalbesselj.(1,x) 
        f_out(x) = im .* y.(x) .* sphericalhankelj.(0,im.*y.(x))./ sphericalhankelj.(1,im.*y.(x))

        xx = collect(range(sqrt(r0[i])-3.5,sqrt(r0[i])-1e-1, length=1000))
        xr1 = xx[convert.(Int64,argmin(abs.(f_in.(xx) .- real.(f_out.(xx)))))]
        xx[convert.(Int64,argmin(abs.(f_in.(xx) .- real.(f_out.(xx)))))] = xx[end];
        xr2 = xx[convert.(Int64,argmin(abs.(f_in.(xx) .- real.(f_out.(xx)))))]
        xr = max(xr1,xr2)

        E_q[i] = (xr/R₀[i])^2*ħ^2/(2*mₑ) + V0[i]
        k₀ = sqrt.(2*mₑ*(E_q[i].-V0[i])/ħ^2)    
        q₀ = sqrt.(-2*mₑ*E_q[i]/ħ^2)


        # Define the integration ranges: r1 and r2 defines the inner and outer regions of the quantum dot
        r1 = range(0.00000000001, R₀[i], length=1000);
        r2 = range(R₀[i], 6.0*10^-9, length=1000);
        z₁ = r1.*k₀; z₂ = r2.*q₀;

        # Calculate the wavefunctions
        R₁ = sin.(z₁) ./ z₁.^2 .- cos.(z₁) ./ z₁; # Inner region
        R₂ = im.*exp.(.-z₂) .* (z₂ .+ 1) ./z₂.^2     ; # Outer region
        
        # Calculate the integrals
        a = integrate(r1, abs.(R₁.*r1).^2) # Inner region
        b = integrate(r2, abs.(R₂.*r2).^2) # Outer region

        # Calculate the normalization constant
        B_cnst = abs.(sin(k₀*R₀[i])/(k₀*R₀[i])^2 - cos(k₀*R₀[i])/(k₀*R₀[i]))^2 / abs(exp(-q₀*R₀[i]) * (q₀ * R₀[i]+1)/(q₀*R₀[i])^2)^2
        
        # Calculate the scaling factors
        A₀ = sqrt(16.0/(3*pi)) *1.0 / sqrt(a + b*B_cnst)
        B₀ = -im*A₀*sqrt(B_cnst)
        t[i] = B₀*integrate(r2, R₂.*(r2).^2)
    end

    return t
end

# Electronic coupling calculation. This code is used to calculate the electronic coupling energy at varius QD sizes.
# This code include the correction for the 1P electrons
function t_size_1P_2(D00::Vector{Float64})
    # Constants
    ħ = 1.054e-34;  # Planck's constant
    m = 9.10938356e-31;  # Electron mass

    # Quantum dot size
    R₀ = D00 .* 1e-9 ./2 # # Quantum dot sizes (diameter) in nanometers
    # Effective mass of electron
    mₑ = 0.13 * m;

    # Electron charge:
    q_e = 1.602176565e-19

    # Define initial potential energy
    V0 = cond_bnd_edge.(D00).*q_e ;

    # Bound state criterion
    r0 =  -2*mₑ*V0 .* R₀ .^2 ./ ħ^2

    x1 = range(1, 5, length=1000)


    E_q = zeros(length(V0)); #Quantized values for the Energy
    # Set titles for the plots
    t = zeros(length(V0))
    for i in eachindex(V0)
        # Start a new local scope
        y1 = sqrt.(abs.(r0[i] .- x1.^2))
        
        ξ1 = x1[findmin(abs.( y1.^2 .*x1.*cos.(x1) .- (x1.^2 .*(1.0 .+y1).+y1.^2).*sin.(x1) ))[2]]


        E_q[i] = (ξ1/R₀[i])^2*ħ^2/(2*mₑ) + V0[i]
        k₀ = sqrt.(2*mₑ*(E_q[i].-V0[i])/ħ^2)    
        q₀ = sqrt.(-2*mₑ*E_q[i]/ħ^2)


        # Define the integration ranges: r1 and r2 defines the inner and outer regions of the quantum dot
        r1 = range(0.00000000001, R₀[i], length=1000);
        r2 = range(R₀[i], 6.0*10^-9, length=1000);
        z₁ = r1.*k₀; z₂ = r2.*q₀;

        # Calculate the wavefunctions
        R₁ = sin.(z₁) ./ z₁.^2 .- cos.(z₁) ./ z₁; # Inner region
        R₂ = im.*exp.(.-z₂) .* (z₂ .+ 1) ./z₂.^2     ; # Outer region
        
        # Calculate the integrals
        a = integrate(r1, abs.(R₁.*r1).^2) # Inner region
        b = integrate(r2, abs.(R₂.*r2).^2) # Outer region

        # Calculate the normalization constant
        B_cnst = abs.(sin(k₀*R₀[i])/(k₀*R₀[i])^2 - cos(k₀*R₀[i])/(k₀*R₀[i]))^2 / abs(exp(-q₀*R₀[i]) * (q₀ * R₀[i]+1)/(q₀*R₀[i])^2)^2
        
        # Calculate the scaling factors
        A₀ = sqrt(16.0/(3*pi)) *1.0 / sqrt(a + b*B_cnst)
        B₀ = -im*A₀*sqrt(B_cnst)
        t[i] = B₀*integrate(r2, R₂.*(r2).^2)
    end

    return t
end


# Plot the electronic coupling energy as a function of the quantum dot size.
function ψ_size_plot_1S()
    # Constants
    ħ = 1.054e-34;  # Planck's constant
    m = 9.10938356e-31;  # Electron mass

    # Quantum dot size
    R₀ = [2.6e-9, 3.0e-9, 3.8e-9, 5.6e-9]./2; # Quantum dot sizes in meters
    R00 = [2.6, 3.0, 3.8, 5.6] # Quantum dot sizes (diameter) in nanometers
    # Effective mass of electron
    mₑ = 0.13 * m;

    # Electron charge:
    q_e = 1.602176565e-19

    # Define initial potential energy
    V0 = cond_bnd_edge.(R00).*q_e ;

    # Initialize plots
    p = plot()
    p1 = plot()

    # Set titles for the plots
    title!(p,"Criterion"); title!(p1,"Probability")
    lbls = ["=4.65×10⁸","=3.28×10⁸","=1.77×10⁸","=6.1×10⁷"]

    # Bound state criterion
    r0 = -2*mₑ*V0.*R₀.^2/ħ^2

    # Start a new local scope
    f(x, y) = x^2 + y^2
    x_a = zeros(length(V0)); y_a = zeros(length(V0))
    x = range(0.001,15, length=1000)
    y = range(0.001,15, length=1000)
    z = @. f(x', y)
    # Plot the criterion function
    plot!(p,x,0.0 .- x.*cot.(x),lw=2,
        xlims = (0,15), ylims = (0,15), labels =  false
    )
    contour!(p, x, y, z, levels=r0, color=:turbo,
        clabels=true, cbar=false, lw=1
    )
    for i in 1:length(V0)
        xx = range(2,3, length=1000)
        x_a[i] = xx[convert.(Int64,argmin(abs.(map(x1->x1^2 + x1^2*(cot(x1))^2-r0[i],xx))))]
        y_a[i] = -x_a[i]*cot(x_a[i])
    end 

    scatter!(p,(x_a,y_a),mc=:red,ms=5,label=false)
    q = y_a./R₀;
    k₀ = x_a./R₀;
    xlabel!(p,L"$k_0 R_0$");
    ylabel!(p,L"$q R_0$");
    # display(p)

    # Set labels for the plots
    # Initialize plots
    p1 = plot()
    p2 = plot()

    # Define colors and labels
    clrs = [:blue :red :green :purple];
    QD_label = ["CdSe 525", "CdSe 550", "CdSe 580", "CdSe 620"]

    x_axis = zeros(2000,4)
    psi_y = zeros(2000,4)

    shell_dot = zeros(4,2)
    
    for i in eachindex(V0)
        # Define the integration ranges: r1 and r2 defines the inner and outer regions of the quantum dot
        r1 = range(0.00000000001, R₀[i], length=1000);
        r2 = range(R₀[i], 6.0*10^-9, length=1000);

        # Calculate the wavefunctions
        R₁ = sin.(k₀[i] * r1) ./ r1; # Inner region
        R₂ = exp.(-q[i] * r2) ./ r2; # Outer region

        # Calculate the integrals
        a = integrate(r1, abs.(R₁.*r1).^2) # Inner region
        b = integrate(r2, abs.(R₂.*r2).^2) # Outer region

        # Calculate the normalization constant
        A_cnst = exp(-2*q[i]*R₀[i]) / (sin(k₀[i] * R₀[i])^2)
        
        # Calculate the scaling factors
        B₀ = 1.0 / sqrt(a*A_cnst + b)
        A₀ = B₀*sqrt(A_cnst)

        # Plot the probability density squared for the inner region
        plot!(p1, r1, A₀ .* R₁ .*(r1).^2,
            lw=2,
            lc=clrs[i],
            label = L"$r_{%$(i)}$= %$(R00[i]) nm"
        )
        # Plot the probability density squared for the outer region
        plot!(p1, r2, (B₀*R₂) .* (r2).^2,
            lw=2,
            lc=clrs[i],
            label=false
        )
        # Add a vertical line at the quantum dot radius
        vline!(p1,(0,R₀[i]),lc=:black,lw=0.5,ls=:dashdot,label=false)
        
        # Plot the radial distribution for the inner region
        plot!(p2, r1.*10^9,  abs.(A₀ * R₁ .* r1).^2*1e-9,
            lw=3,
            lc=clrs[i],
            label = QD_label[i]
        )
        # Plot the radial distribution for the outer region
        plot!(p2, r2.*10^9, abs.(B₀ * R₂ .* r2).^2*1e-9,
            lw=3,
            lc=clrs[i],
            label=false
        )
        x_axis[:,i] = vcat(r1.*10^9,r2.*10^9)
        psi_y[:,i] = vcat((A₀ * R₁ .* r1).^2*1e-9,(B₀ * R₂ .* r2).^2*1e-9)
        # Add a scatter plot point at the quantum dot radius
        tmp=abs.(A₀ * R₁ .*r1)[end]^2
        scatter!(p2, (R₀[i].*10^9,tmp*1e-9),
                mc=clrs[i],
                yerr = [0.0],
                ma=1,
                ms = 7,
                label=false
            )
        shell_dot[i,1] = R₀[i].*10^9
        shell_dot[i,2] = tmp*1e-9
        # Add a vertical line at the quantum dot radius
        vline!(p2,(0,R₀[i].*10^9),lc=clrs[i],lw=0.5,ls=:dashdot,label=false)
    end 

    # Set labels and title for the first plot
    plot!(p1, xlabel=L"$r$ (m)", ylabel=L"$|\psi(r)|^2$", title="Probability",
        legend=:outertop,
        legendcolumns=4
    )
    xlims!(p1,0, 5e-9)
    plot!(p1,legendfontsize = 9, size = (800,600)) # Set the font size of the labels to 10 and the legend to 8
    # display(p1)
    

    # Set labels and title for the second plot
    plot!(p2, xlabel=L"$r$ (nm)", ylabel=L"$r^2|\psi(r)|^2$", title = "1S radial distribution",
        titlefontsize=18
        )
    xlims!(p2,0, 3.4)
    ylims!(p2,0, 1.2)
    annotate!(p2, -0.19, 1.23, text("×10⁹", 11, :left))
    plot!(p2,legendfontsize = 12, size = (800,600), xguidefontsize=18, yguidefontsize=20,
        xtickfontsize=12, ytickfontsize = 12, grid = false,
        left_margin=4Plots.mm
        ) # Set the font size of the labels to 10 and the legend to 8
    df = DataFrame(R_size_1S = vec(x_axis), psi_size_1S = vec(psi_y))

    # Write the DataFrame to a CSV file
    CSV.write("wavefunction_size_1S.csv", df)
    println(shell_dot)
    return p2
end

# Plot the electronic coupling energy as a function of the quantum dot size.
function ψ_size_plot_1P()
    # Define colors and labels
    clrs = [:blue :red :green :purple];
    QD_label = ["CdSe 525", "CdSe 550", "CdSe 580", "CdSe 620"]

    # Constants
    ħ = 1.054e-34;  # Planck's constant
    m = 9.10938356e-31;  # Electron mass

    # Quantum dot size
    R₀  = [2.6e-9, 3.0e-9, 3.8e-9, 5.6e-9]./2; # Quantum dot sizes in meters
    R00 = [2.6, 3.0, 3.8, 5.6] # Quantum dot sizes (diameter) in nanometers
    # Effective mass of electron
    mₑ = 0.13 * m;

    # Electron charge:
    q_e = 1.602176565e-19

    # Define initial potential energy
    V0 = cond_bnd_edge.(R00).*q_e ;

    x1 = range(1, 5, length=1000)

    # Bound state criterion
    r0 =  -2*mₑ*V0 .* R₀ .^2 ./ ħ^2

    # Initialize plots
    p1 = plot()

    E_q = zeros(length(V0)); #Quantized values for the Energy

    x_axis = zeros(2000,4)
    psi_y = zeros(2000,4)

    shell_dot = zeros(4,2)

    for i in eachindex(V0)
        # Start a new local scope
        y1 = sqrt.(abs.(r0[i] .- x1.^2))
        
        ξ1 = x1[findmin(abs.( y1.^2 .*x1.*cos.(x1) .- (x1.^2 .*(1.0 .+y1).+y1.^2).*sin.(x1) ))[2]]


        E_q[i] = (ξ1/R₀[i])^2*ħ^2/(2*mₑ) + V0[i]
        k₀ = sqrt.(2*mₑ*(E_q[i].-V0[i])/ħ^2)    
        q₀ = sqrt.(-2*mₑ*E_q[i]/ħ^2)


        # Define the integration ranges: r1 and r2 defines the inner and outer regions of the quantum dot
        r1 = range(0.00000000001, R₀[i], length=1000);
        r2 = range(R₀[i], 6.0*10^-9, length=1000);
        z₁ = r1.*k₀; z₂ = r2.*q₀;

        # Calculate the wavefunctions
        R₁ = sin.(z₁) ./ z₁.^2 .- cos.(z₁) ./ z₁; # Inner region
        R₂ = im.*exp.(.-z₂) .* (z₂ .+ 1) ./z₂.^2     ; # Outer region
        
        # Calculate the integrals
        a = integrate(r1, abs.(R₁.*r1).^2) # Inner region
        b = integrate(r2, abs.(R₂.*r2).^2) # Outer region

        # Calculate the normalization constant
        B_cnst = abs.(sin(k₀*R₀[i])/(k₀*R₀[i])^2 - cos(k₀*R₀[i])/(k₀*R₀[i]))^2 / abs(exp(-q₀*R₀[i]) * (q₀ * R₀[i]+1)/(q₀*R₀[i])^2)^2
        
        # Calculate the scaling factors
        A₀ = sqrt(16.0/(3*pi)) *1.0 / sqrt(a + b*B_cnst)
        B₀ = -im*A₀*sqrt(B_cnst)

                # Plot the radial distribution for the inner region
        plot!(p1, r1.*10^9,  abs.(A₀ * R₁ .* r1).^2*1e-9,
            lw=3,
            lc=clrs[i],
            label = QD_label[i]
        )
        # Plot the radial distribution for the outer region
        plot!(p1, r2.*10^9, abs.(B₀ * R₂ .* r2).^2*1e-9,
            lw=3,
            lc=clrs[i],
            label=false
        )
        x_axis[:,i] = vcat(r1.*10^9,r2.*10^9)
        psi_y[:,i] = vcat((A₀ * R₁ .* r1).^2*1e-9,(B₀ * R₂ .* r2).^2*1e-9)
        # Add a scatter plot point at the quantum dot radius
        tmp=(A₀ * R₁ .* r1)[end]^2
        scatter!(p1, (R₀[i].*10^9,tmp*1e-9),
                mc=clrs[i],
                yerr = [0.0],
                ma=1,
                ms = 7,
                label=false
            )
        # Add a vertical line at the quantum dot radius
        vline!(p1,(0,R₀[i].*10^9),lc=clrs[i],lw=0.5,ls=:dashdot,label=false)
        shell_dot[i,1] = R₀[i].*10^9
        shell_dot[i,2] = tmp*1e-9
    end

    # Set labels and title for the second plot
    plot!(p1, xlabel=L"$r$ (nm)", ylabel=L"$r^2|\psi(r)|^2$", title = "1P radial distribution",
        titlefontsize=18
        )
    xlims!(p1,0, 4)
    ylims!(p1,0, 2.2)
    # ylims!(p2,0, 1.2)
    annotate!(p1, -0.21, 2.27, text("×10⁹", 11, :left))
    plot!(p1,legendfontsize = 12, size = (800,600), xguidefontsize=18, yguidefontsize=20,
        xtickfontsize=12, ytickfontsize = 12, grid = false,
        left_margin=4Plots.mm
        ) # Set the font size of the labels to 10 and the legend to 8
    
    df = DataFrame(R_size_1P = vec(x_axis), psi_size_1P = vec(psi_y))

    # Write the DataFrame to a CSV file
    CSV.write("wavefunction_size_1P.csv", df)

    println(shell_dot)
    return p1;
end

# Plot the electronic coupling energy as a function of the quantum dot size.
function ψ_size_plot_1P_2()
    # Constants
    ħ = 1.054e-34;  # Planck's constant
    m = 9.10938356e-31;  # Electron mass

    # Quantum dot size
    R₀  = [2.6e-9, 3.0e-9, 3.8e-9, 5.6e-9]./2; # Quantum dot sizes in meters
    R00 = [2.6, 3.0, 3.8, 5.6] # Quantum dot sizes (diameter) in nanometers
    # Effective mass of electron
    mₑ = 0.13 * m;

    # Electron charge:
    q_e = 1.602176565e-19

    # Define initial potential energy
    V0 = cond_bnd_edge.(R00).*q_e ;

    x1 = range(1, 5, length=1000)

    # Bound state criterion
    r0 =  -2*mₑ*V0 .* R₀ .^2 ./ ħ^2

    # Definition of the spherical Bessel and Hankel functions
    sphericalhankelj(ν, x) = √(π/2x).*hankelh1(ν+1/2, x)
    sphericalbesselj(ν, x) = √(π/2x).*besselj(ν+1/2, x)

    # Initialize plots
    p1 = plot()
    p2 = plot()
    p3 = plot()
    p4 = plot()

    p = [p1, p2, p3, p4]
    E_q = zeros(length(V0)); #Quantized values for the Energy
    # Set titles for the plots
    lbls = ["=4.65×10⁸","=3.28×10⁸","=1.77×10⁸","=6.1×10⁷"]
    
    for i in eachindex(V0)
        # Start a new local scope
        # q(E) = sqrt.(2*mₑ*E/ħ^2)
        # k(E) = sqrt.(2*mₑ*(E.-V0[i])/ħ^2)

        y(E) = sqrt.(r0[i] .- E.^2)

        f_in(E) = E .* sphericalbesselj.(0,E)./ sphericalbesselj.(1,E) 
        f_out(E) = im .* y.(E) .* sphericalhankelj.(0,im.*y.(E))./ sphericalhankelj.(1,im.*y.(E))

        xx = collect(range(sqrt(r0[i])-3.5,sqrt(r0[i])-1e-1, length=10000))
        xr1 = xx[convert.(Int64,argmin(abs.(f_in.(xx) .- real.(f_out.(xx)))))]
        xx[convert.(Int64,argmin(abs.(f_in.(xx) .- real.(f_out.(xx)))))] = xx[end];
        xr2 = xx[convert.(Int64,argmin(abs.(f_in.(xx) .- real.(f_out.(xx)))))]
        xr = max(xr1,xr2)
        E_q[i] = (xr/R₀[i])^2*ħ^2/(2*mₑ) + V0[i]

        # f_in(E) = k.(E)*R₀[i].*(sin.(k(E) *R₀[i]).*(k.(E)*R₀[i] .- 
        #     2.0./(k.(E)*R₀[i]) ) .+ 2.0.*cos.(k.(E)*R₀[i]))./
        #     (sin.(k.(E)*R₀[i]) .- k.(E).*R₀[i].*cos.(k.(E)*R₀[i]))
        # f_out(E) = q(E)*R₀[i].*(1.0 .+ 2.0./(q(E)*R₀[i]) .- 1.0/(1.0 .- q(E)*R₀[i]))

        # Plot the criterion function
        plot!(p[i],-xx,f_in.(xx),lw=2, ylims = (-15,15), labels =  false
        )
        plot!(p[i],-xx,real.(f_out.(xx)),lw=2, ylims = (-15,15), labels =  false
        )
        plot!(p[i],xlabel=L"Energy $E$", ylabel=L"$f(E)$", title="Criterion",
        )
        # vline!(p[i],[-r0[i]], labels = false)
        scatter!(p[i],[-xr],[f_in(xr)],mc=:red,ms=5,label=false)
        # display(p[i])
    end
    k₀ = sqrt.(2*mₑ*(E_q.-V0)/ħ^2)    
    q₀ = sqrt.(-2*mₑ*E_q/ħ^2)

    # Set labels for the plots
    # Initialize plots
    p1 = plot()
    p2 = plot()

    # Define colors and labels
    clrs = [:blue :red :green :purple];
    QD_label = ["CdSe 525", "CdSe 550", "CdSe 580", "CdSe 620"]
    
    x_axis = zeros(2000,4)
    psi_y = zeros(2000,4)

    shell_dot = zeros(4,2)
    
    for i in eachindex(V0)
        # Define the integration ranges: r1 and r2 defines the inner and outer regions of the quantum dot
        r1 = range(0.00000000001, R₀[i], length=1000);
        r2 = range(R₀[i], 6.0*10^-9, length=1000);
        z₁ = r1.*k₀[i]; z₂ = r2.*q₀[i];

        # Calculate the wavefunctions
        R₁ = sin.(z₁) ./ z₁.^2 .- cos.(z₁) ./ z₁; # Inner region
        R₂ = im.*exp.(.-z₂) .* (z₂ .+ 1) ./z₂.^2     ; # Outer region
        
        # Calculate the integrals
        a = integrate(r1, abs.(R₁.*r1).^2) # Inner region
        b = integrate(r2, abs.(R₂.*r2).^2) # Outer region

        # Calculate the normalization constant
        B_cnst = abs.(sin(k₀[i]*R₀[i])/(k₀[i]*R₀[i])^2 - cos(k₀[i]*R₀[i])/(k₀[i]*R₀[i]))^2 / abs(exp(-q₀[i]*R₀[i]) * (q₀[i] * R₀[i]+1)/(q₀[i]*R₀[i])^2)^2
        
        # Calculate the scaling factors
        A₀ = sqrt(16.0/(3*pi)) *1.0 / sqrt(a + b*B_cnst)
        B₀ = A₀*sqrt(B_cnst)

        # Plot the probability density squared for the inner region
        plot!(p1, r1, (A₀*R₁).^2,
            lw=2,
            lc=clrs[i],
            label = L"$r_{%$(i)}$= %$(R00[i]) nm"
        )
        # Plot the probability density squared for the outer region
        plot!(p1, r2, abs.(B₀*R₂).^2,
            lw=2,
            lc=clrs[i],
            label=false
        )
        # Add a vertical line at the quantum dot radius
        vline!(p1,(0,R₀[i]),lc=:black,lw=0.5,ls=:dashdot,label=false)
        
        # Plot the radial distribution for the inner region
        plot!(p2, r1.*10^9,  abs.(A₀ * R₁ .* r1).^2*1e-9,
            lw=3,
            lc=clrs[i],
            label = QD_label[i]
        )
        # Plot the radial distribution for the outer region
        plot!(p2, r2.*10^9, abs.(B₀ * R₂ .* r2).^2*1e-9,
            lw=3,
            lc=clrs[i],
            label=false
        )
        x_axis[:,i] = vcat(r1.*10^9,r2.*10^9)
        psi_y[:,i] = vcat((A₀ * R₁ .* r1).^2*1e-9,(B₀ * R₂ .* r2).^2*1e-9)
        # Add a scatter plot point at the quantum dot radius
        tmp=(A₀ * R₁ .* r1)[end]^2
        scatter!(p2, (R₀[i].*10^9,tmp*1e-9),
                mc=clrs[i],
                yerr = [0.0],
                ma=1,
                ms = 7,
                label=false
            )
        shell_dot[i,1] = R₀[i].*10^9
        shell_dot[i,2] = tmp*1e-9
        # Add a vertical line at the quantum dot radius
        vline!(p2,(0,R₀[i].*10^9),lc=clrs[i],lw=0.5,ls=:dashdot,label=false)
    end 

    # Set labels and title for the first plot
    plot!(p1, xlabel=L"$r$ (m)", ylabel=L"$|\psi(r)|^2$", title="Probability",
        legend=:outertop,
        legendcolumns=4
    )
    xlims!(p1,0, 4e-9)
    plot!(p1,legendfontsize = 9, size = (800,600)) # Set the font size of the labels to 10 and the legend to 8
    # display(p1)
    

    # Set labels and title for the second plot
    plot!(p2, xlabel=L"$r$ (nm)", ylabel=L"$r^2|\psi(r)|^2$", title = "1P radial distribution",
        titlefontsize=18
        )
    xlims!(p2,0, 4)
    ylims!(p2,0, 2.2)
    # ylims!(p2,0, 1.2)
    annotate!(p2, -0.21, 2.27, text("×10⁹", 11, :left))
    plot!(p2,legendfontsize = 12, size = (800,600), xguidefontsize=18, yguidefontsize=20,
        xtickfontsize=12, ytickfontsize = 12, grid = false,
        left_margin=4Plots.mm
        ) # Set the font size of the labels to 10 and the legend to 8
    
    df = DataFrame(R_size_1P = vec(x_axis), psi_size_1P = vec(psi_y))

    # Write the DataFrame to a CSV file
    CSV.write("wavefunction_size_1P.csv", df)
    return p2;
end


########################################################################
########## Codes for the electronic coupling calculation ###############
##########        for the distance dependance            ###############
########################################################################

function t_dis_1S(D00::Vector{Float64})
    # Constants
    ħ = 1.054e-34;  # Planck's constant
    m = 9.10938356e-31;  # Electron mass

    # Quantum dot size
    R₀ = 3.15e-9/2;  # Quantum dot sizes in meters
    R₁ = R₀ .+ 0.5.*D00.*10^-9; # Shell thickness
    # R₁ = R₀ .+ [0.64, 0.77, 1.75].*10^-9; # Shell thickness

    # Effective mass of electron
    mₑ1 = 0.13 * m;
    mₑ2 = 0.25 * m; 

    # Electron charge:
    q_e = 1.602176565e-19

    # Define initial potential energy
    V1 = cond_bnd_edge((2*R₀)*10^9)*q_e ;
    # V1 = -4.0*q_e;
    V2 = V1 + 1.4*q_e;

    t_coupling = zeros(length(R₁));


    # # Calculation for the CdSe/ZnS core/shell QD


    # Bound states criterion
    r0 = 2 * R₀^2 * (V2-V1) / ħ^2

    xx = range(1.5, 2.5, length=5000)
    x_a= xx[convert.(Int64, argmin(abs.(map(x1 -> x1^2/ mₑ1 + mₑ2 * (x1 * cot(x1)/ mₑ1 + 1/mₑ2 - 1/mₑ1 )^2 - r0, xx))))]
    #  y_a= -mₑ2 * x_a * cot(x_a)/mₑ1 + mₑ2/mₑ1 - 1
    y_a= sqrt(mₑ2* (r0-x_a^2/mₑ1 ) )

    q₂ = y_a / R₀;
    k₁ = x_a / R₀


    En = ħ^2 * k₁^2 / (2 * mₑ1) + V1
    q₃ = sqrt(-2 * mₑ1 * En / ħ^2)
    for i in 1:length(R₁)
        r1 = range(0.000000000001, R₀, length=2000);
        r2 = range(R₀, R₁[i]  , length=1000);
        r3 = range(R₁[i], 6.0e-9, length=1000);

        u₁ = sin.(k₁ * r1) ./ r1;
        u₂ = exp.(-q₂ * r2) ./ r2;
        u₃ = exp.(-q₃ * r3) ./ r3;
        a = integrate(r1, abs.(u₁.*r1).^2)
        b = integrate(r2, abs.(u₂.*r2).^2)
        cc = integrate(r3, abs.(u₃.*r3).^2)
        A_cnst = exp(-2*q₂*R₀) / (sin(k₁ * R₀)^2)
        C_cnst = exp(2*R₁[i]*(q₃ - q₂))

        # #
        B₀ = 1.0 / sqrt(a*A_cnst + b + cc*C_cnst);
        C₀ = B₀*sqrt(C_cnst);
        t_coupling[i] = C₀*integrate(r3,u₃.*r3.^2)
    end

    return t_coupling
end


# Electronic coupling calculation. This code is used to calculate the electronic coupling energy at varius QD Core/Shell thickness.
function t_dis_1P(D00::Vector{Float64})
    # Constants
    ħ = 1.054e-34;  # Planck's constant
    m = 9.10938356e-31;  # Electron mass
 
    # Quantum dot size
    R₀   = 3.154e-9/2;  # Quantum dot sizes in meters
    R_cs = R₀ .+ 0.5.*D00.*10^-9; # Shell thickness
    # R₀ = 2.74e-9/2;  
    # R₁ = R₀ + 1e-9;
    # Effective mass of electron
    mₑ1 = 0.13 * m;
    mₑ2 = 0.25 * m; 
 
    # Electron charge:
    q_e = 1.602176565e-19
 
    # Define initial potential energy
    V1 = cond_bnd_edge((2*R₀)*10^9)*q_e ;
    V2 = V1 + 1.4*q_e;

    # Definition of the spherical Bessel and Hankel functions
    sphericalhankelj(ν, x) = √(π/2x).*hankelh1(ν+1/2, x)
    sphericalbesselj(ν, x) = √(π/2x).*besselj(ν+1/2, x)

    t_coupling = zeros(length(R_cs));

    # # # calculation of the QD with the shell

     # Bound states criterion
     r0 = 2 * R₀^2 * (V2-V1) / ħ^2
     
     # Start a new local scope
     yy(E) = sqrt.(r0*mₑ2 .- E.^2.0 .*(mₑ2/mₑ1))

     ff_in(E) = E .* sphericalbesselj.(0,E)./ (mₑ1 .* sphericalbesselj.(1,E) ) .- 2/mₑ1 
     ff_out(E) = im .* yy.(E) .* sphericalhankelj.(0,im.*yy.(E))./ (mₑ2 .* sphericalhankelj.(1,im.*yy.(E)) ) .- 2/mₑ2
     
     xx  = range(sqrt(r0*mₑ1)-3.4,sqrt(r0*mₑ1)-1e-2, length=1000)
     xr₁ = xx[convert.(Int64,argmin(abs.(ff_in.(xx) .- real.(ff_out.(xx)))))]
     k₁  = xr₁/R₀
     q₂  = sqrt.(r0*mₑ2 .- xr₁^2*(mₑ2/mₑ1))/R₀
     xr₂ = q₂*R₀
     En  = (xr₁/R₀)^2*ħ^2/(2*mₑ1) + V1



     q₃   = sqrt(-2 * mₑ1 * En / ħ^2)
     xr₂₃ = q₂ .* R_cs
     xr₃  = q₃ .* R_cs
     for i in eachindex(R_cs)
         r1 = range(0.0000000000001, R₀, length=2000);
         r2 = range(R₀, R_cs[i], length=1000);
         r3 = range(R_cs[i], 6.0e-9, length=1000);
         z₁ = r1.*k₁; z₂ = im .* r2.*q₂; z₃ = im .* r3.*q₃;
 
         # Calculate the wavefunctions
         R₁ = sphericalbesselj.(1,z₁); # Inner region
         R₂ = im*imag.(sphericalhankelj.(1,z₂)); # Outer region
         R₃ = im*imag.(sphericalhankelj.(1,z₃)); # Outer region
         a  = integrate(r1, abs.(R₁.*r1).^2)
         b  = integrate(r2, abs.(R₂.*r2).^2)
         cc = integrate(r3, abs.(R₃.*r3).^2)
         B_cnst = -im.*sphericalbesselj.(1,xr₁) / imag.(sphericalhankelj.(1,im*xr₂))
         C_cnst = -im.*sphericalbesselj.(1,xr₁) * imag.(sphericalhankelj.(1,im*xr₂₃[i])) / (imag.(sphericalhankelj.(1,im*xr₃[i])) * imag.(sphericalhankelj.(1,im*xr₂))) 
 
         # #
         A₀ = sqrt(16.0/(3*pi)) *1.0 / sqrt(a + b*abs.(B_cnst)^2 + cc*abs.(C_cnst)^2);
         B₀ = im*imag(A₀*B_cnst);
         C₀ = im*imag(A₀*C_cnst);
        
         t_coupling[i] = C₀*integrate(r3,R₃.*r3.^2)
    end

    return t_coupling
end


function ψ_dis_plot_1S()
    # Constants
    ħ = 1.054e-34;  # Planck's constant
    m = 9.10938356e-31;  # Electron mass
 
    # Quantum dot size
    R₀ = 3.154e-9/2;  # Quantum dot sizes in meters
    R₁ = R₀ .+ 0.5.*[0.64, 0.77, 1.75].*10^-9; # Shell thickness
    # R₀ = 2.74e-9/2;  
    # R₁ = R₀ + 1e-9;
 
    # Effective mass of electron
    mₑ1 = 0.13 * m;
    mₑ2 = 0.25 * m; 
 
    # Electron charge:
    q_e = 1.602176565e-19
 
    # Define initial potential energy
    # V1 = -4.0*q_e;
    # V2 = -3.1*q_e;
    V1 = cond_bnd_edge((2*R₀)*10^9)*q_e ;
    V2 = V1 + 1.4*q_e;
 
    # Initialize plots
    p1 = plot()
    p2 = plot()
    p3 = plot()
    p4 = plot()
    # Set titles for the plots
    title!(p1, "Criterion");
    title!(p3, "Probability")
    lbls = ["=2.88×10⁸", "=2.46×10⁸","=1.96×10⁸","=3.7×10⁷"]
    # title!(p4, "Radial distribution")

    clrs = [:blue :red :green :purple];
    QD_label    = ["CdSe - Core", "CdSe/ZnS₀.₆₄", "CdSe/ZnS₀.₇₇", "CdSe/ZnS₁.₇₅"]

    x_axis = zeros(4000,4)
    psi_y = zeros(4000,4)

    shell_dot = zeros(4,2)

    # # Calculation for the Bare QD, i.e., no Shell
    r0 = -2*mₑ1*V1*R₀^2/ħ^2     

    # Start a new local scope
    ff(x, y) = x^2 + y^2
    x_a = 0.0; y_a = 0.0
    x = range(0.001,10, length=1000)
    y = range(0.001,10, length=1000)
    z = @. ff(x', y)
    # Plot the criterion function
    plot!(p1, x, 0.0 .- x .* cot.(x), lw=2,
        xlims = (0, 10), ylims = (0, 10), labels =  false
    )
    contour!(p1, x, y, z, levels=[r0], color=:turbo,
        clabels=true, cbar=false, lw=1
    )
    xx = range(2,3, length=1000)
    x_a = xx[convert.(Int64,argmin(abs.(map(x1->x1^2 + x1^2*(cot(x1))^2-r0,xx))))]
    y_a = -x_a*cot(x_a)

    scatter!(p1, (x_a, y_a), mc=:red, ms=5, label=false)
    xlabel!(p1, L"$k_0 R_0$");
    ylabel!(p1, L"$q_0 R_0$");
    q = y_a./R₀;
    k₀ = x_a./R₀;
    #  println("k₀ = $k₀")

    r1 = range(0.000000000001,R₀, length=2000);
    r2 = range(R₀,6.0*10^-9, length=2000);

    u₁ = sin.(k₀ * r1) ./ r1;
    u₂ = exp.(-q * r2) ./ r2;

    a = integrate(r1, abs.(u₁.*r1).^2)
    b = integrate(r2, abs.(u₂.*r2).^2)

    A_cnst = exp(-2*q*R₀) / (sin(k₀ * R₀)^2)

    B₀ = 1.0 / sqrt(a*A_cnst + b)
    A₀ = B₀*sqrt(A_cnst)
    plot!(p3, r1, (abs.(A₀*u₁).^2),
            lw=2,
            lc=clrs[1],
            label=false
        )
    plot!(p3, r2, (abs.(B₀*u₂).^2),
            lw=2,
            lc=clrs[1],
            label=false
        )
    plot!(p4, r1.*10^9, (A₀ * u₁ .* r1).^2*1e-9,
        lw=3,
        lc=clrs[1],
        label=false
    )
    plot!(p4, r2.*10^9,(B₀ * u₂ .* r2).^2*1e-9,
        lw=3,
        lc=clrs[1],
        label=false #QD_label[1]
    )
    vline!(p3, (0, R₀), lc=clrs[1], lw=0.5, ls=:dashdot, label=false)
    vline!(p4, (0, R₀.*10^9), lc=clrs[1], lw=0.5, ls=:dashdot, label=false)
    
    # Only core output data
    shell_dot[1,1] = R₀.*10^9
    shell_dot[1,2] = (A₀ * u₁[end] .* r1[end]).^2*1e-9
    #Continuum line
    x_axis[:,1] = vcat(r1.*10^9,r2.*10^9)
    psi_y[:,1] = vcat((A₀ * u₁ .* r1).^2*1e-9,(B₀ * u₂ .* r2).^2*1e-9)

    # scatter!(p4, (R₀.*10^9,tmp*1e-9),
    #     mc=clrs[1],
    #     yerr = [0.0],
    #     ma=1,
    #     ms = 7,
    #     # label=L"$R_{1}^2|\psi(R_{1})|^2$= %$(string(round(tmp/(10 ^(floor.(log.(10,tmp)))),digits=2)))×10\^{%$(convert(Int64,floor(log(10,tmp))))}"
    #     label=false; # L"$R_{1}^2|\psi(R_{1})|^2$"*lbls[1]
    # )

    # # calculation of the QD with the shell

    # Bound states criterion
    r0 = 2 * R₀^2 * (V2-V1) / ħ^2

    # Start a new local scope
    f(x, y) = x^2/mₑ1 + y^2/mₑ2
    x_a = 0.0;
    y_a = 0.0;
    x = range(0.001, 10, length=1000)
    y = range(0.001, 10, length=1000)
    z = @. f(x', y)

    # Plot the criterion function
    plot!(p2, x, 0.0 .- mₑ2 .* (x .* cot.(x) ./ mₑ1 .+ 1/mₑ2 .- 1/mₑ1) , lw=2,
        xlims = (0, 10), ylims = (0, 10), labels =  false)
    contour!(p2, x, y, z, levels=[r0], color=:turbo,
        clabels=true, cbar=false, lw=1
    )
    xx = range(1.5, 2.5, length=5000)
    x_a= xx[convert.(Int64, argmin(abs.(map(x1 -> x1^2/ mₑ1 + mₑ2 * (x1 * cot(x1)/ mₑ1 + 1/mₑ2 - 1/mₑ1 )^2 - r0, xx))))]
    #  y_a= -mₑ2*(x_a*cot(x_a)/mₑ1 + 1/mₑ2 - 1/mₑ1)

    y_a= sqrt(mₑ2* (r0-x_a^2/mₑ1 ) )

    scatter!(p2, (x_a, y_a ), mc=:red, ms=5, label=false)
    q₂ = y_a / R₀;
    k₁ = x_a / R₀;
    println("k₁ = $k₁")
    xlabel!(p2, L"$k_1 R_0$");
    ylabel!(p2, L"$q_2 R_0$");
    #  display(plot(p1, p2, layout = grid(2,1)))


    En = ħ^2 * k₁^2 / (2 * mₑ1) + V1
    q₃ = sqrt(-2 * mₑ1 * En / ħ^2)
    
    for i in eachindex(R₁)
        r1 = range(0.0000000000001, R₀, length=2000);
        r2 = range(R₀, R₁[i], length=1000);
        r3 = range(R₁[i], 6.0e-9, length=1000);

        x_axis[:,i+1] = vcat(r1.*10^9,r2.*10^9,r3.*10^9)

        u₁ = sin.(k₁ * r1) ./ r1;
        u₂ = exp.(-q₂ * r2) ./ r2;
        u₃ = exp.(-q₃ * r3) ./ r3;
        a = integrate(r1, abs.(u₁.*r1).^2)
        b = integrate(r2, abs.(u₂.*r2).^2)
        cc = integrate(r3, abs.(u₃.*r3).^2)
        A_cnst = exp(-2*q₂*R₀) / (sin(k₁ * R₀)^2)
        C_cnst = exp(2*R₁[i]*(q₃ - q₂))

        # #
        B₀ = 1.0 / sqrt(a*A_cnst + b + cc*C_cnst);
        A₀ = B₀*exp(-q₂*R₀) / sin(k₁ * R₀);
        C₀ = B₀*exp(R₁[i]*(q₃ - q₂));

        plot!(p3, r1, abs.(A₀*u₁).^2,
            lw=2,
            lc=clrs[i+1],
            label=false
        )
        plot!(p3, r2, abs.(B₀*u₂).^2,
            lw=2,
            lc=clrs[i+1],
            label=false
        )
        plot!(p3, r3, abs.(C₀*u₃).^2,
            lw=2,
            lc=clrs[i+1],
            label=false
        )
        #  println(log.((B₀*u₂.*r2).^2)[1]-log.((B₀*u₂.*r2).^2)[end])
        # #

        plot!(p4, r1.*10^9, (A₀ * u₁ .* r1).^2*1e-9,
            lw=3,
            lc=clrs[i+1],
            label=false #QD_label[i+1]
        )
        plot!(p4, r2.*10^9,  (B₀ * u₂ .* r2).^2*1e-9,
            lw=3,
            lc=clrs[i+1],
            label=false,
        )
        plot!(p4, r3.*10^9,  (C₀ * u₃ .* r3).^2*1e-9,
            lw=3,
            lc=clrs[i+1],
            label=false, ytickfontsize=10,
        )
        psi_y[:,i+1] = vcat((A₀ * u₁ .* r1).^2*1e-9,(B₀ * u₂ .* r2).^2*1e-9,(C₀ * u₃ .* r3).^2*1e-9)
        vline!(p3, (0, R₁[i]), lc=clrs[i+1], lw=0.5, ls=:dashdot, label=false)
        vline!(p4, (0, R₁[i]*10^9), lc=clrs[i+1], lw=0.5, ls=:dashdot, label=false)
    end
    
    for i in eachindex(R₁)
        r1 = range(0.0000000000001, R₀, length=2000);
        r2 = range(R₀, R₁[i], length=1000);
        r3 = range(R₁[i], 6.0e-9, length=1000);

        u₁ = sin.(k₁ * r1) ./ r1;
        u₂ = exp.(-q₂ * r2) ./ r2;
        u₃ = exp.(-q₃ * r3) ./ r3;
        a = integrate(r1, abs.(u₁.*r1).^2)
        b = integrate(r2, abs.(u₂.*r2).^2)
        cc = integrate(r3, abs.(u₃.*r3).^2)
        A_cnst = exp(-2*q₂*R₀) / (sin(k₁ * R₀)^2)
        C_cnst = exp(2*R₁[i]*(q₃ - q₂))

        # #
        B₀ = 1.0 / sqrt(a*A_cnst + b + cc*C_cnst);
        A₀ = B₀*exp(-q₂*R₀) / sin(k₁ * R₀);
        tmp=(B₀ * u₂ .* r2)[end]^2
        scatter!(p4, (R₁[i].*10^9,tmp*1e-9),
                mc=clrs[i+1],
                yerr = [0.0],
                ma=1,
                ms = 7,
                label=false; # L"$R_{%$(i)}^2|\psi(R_{%$(i)})|^2$"*lbls[i+1]
            ) 
        shell_dot[i+1,2] = tmp*1e-9
        shell_dot[i+1,1] = R₁[i].*10^9
    end
    plot!(p3, xlabel=L"$r$ (m)", ylabel=L"$|\psi(r)|$", title="Probability",
         legend=:topleft, legendcolumns=4
     )
    xlims!(p3, 0, 4e-9)
    plot!(p3, legendfontsize = 9, size = (800,600)) # Set the font size of the labels to 10 and the legend to 8
    # display(p3)
 
    plot!(p4, xlabel=L"$r$ (nm)", ylabel=L"$r^2|\psi(r)|^2$",
         legend=:topright, grid=false,
         framestyle = :box # Example: Tick line width
     )
    xlims!(p4, 0, 3.4)
    ylims!(p4,0, 1.2)
    annotate!(p4, -0.19, 1.23, text("×10⁹", 11, :left))
    plot!(p4,legendfontsize = 12, size = (800,600), xguidefontsize=18, yguidefontsize=20,
    xtickfontsize=15, ytickfontsize = 15, grid = false,
    left_margin=4Plots.mm) # Set the font size of the labels to 10 and the legend to 8
    display(p4)

    df = DataFrame(R_dis_1S = vec(x_axis), psi_dis_1S = vec(psi_y))

    # Write the DataFrame to a CSV file
    CSV.write("wavefunction_dis_1S.csv", df)
    return x_axis, psi_y, shell_dot
end
# julia> core_wf, core_radius, scatter_core, shell_wfs, shell_radius, scatter_shell = ψ_dis_plot_1S()

function ψ_dis_plot_1P()
    # Constants
    ħ = 1.054e-34;  # Planck's constant
    m = 9.10938356e-31;  # Electron mass
 
    # Quantum dot size
    R₀   = 3.154e-9/2;  # Quantum dot sizes in meters
    R_cs = R₀ .+ 0.5.*[0.64, 0.77, 1.75].*10^-9; # Shell thickness
    # R₀ = 2.74e-9/2;  
    # R₁ = R₀ + 1e-9;
    # Effective mass of electron
    mₑ1 = 0.13 * m;
    mₑ2 = 0.25 * m; 
 
    # Electron charge:
    q_e = 1.602176565e-19
 
    # Define initial potential energy
    V1 = cond_bnd_edge((2*R₀)*10^9)*q_e ;
    V2 = V1 + 1.4*q_e;

    # Definition of the spherical Bessel and Hankel functions
    sphericalhankelj(ν, x) = √(π/2x).*hankelh1(ν+1/2, x)
    sphericalbesselj(ν, x) = √(π/2x).*besselj(ν+1/2, x)
 
    # Initialize plots
    p1 = plot()
    p2 = plot()
    p3 = plot()
    p4 = plot()
    p = [p1, p2, p3, p4]
    # Set titles for the plots
    title!(p1, "Criterion");
    title!(p3, "Probability")
    lbls = ["=2.88×10⁸", "=2.46×10⁸","=1.96×10⁸","=3.7×10⁷"]
    # title!(p4, "Radial distribution")

    clrs = [:blue :red :green :purple];
    QD_label    = ["CdSe - Core", "CdSe/ZnS₀.₆₄", "CdSe/ZnS₀.₇₇", "CdSe/ZnS₁.₇₅"]

    x_axis = zeros(4000,4)
    psi_y = zeros(4000,4)

    shell_dot = zeros(4,2)
 
    # # Calculation for the Bare QD, i.e., no Shell
     r0 = -2*mₑ1*V1*R₀^2/ħ^2     

     # Start a new local scope
     y(E) = sqrt.(r0 .- E.^2)

     f_in(E) = E .* sphericalbesselj.(0,E)./ sphericalbesselj.(1,E) 
     f_out(E) = im .* y.(E) .* sphericalhankelj.(0,im.*y.(E))./ sphericalhankelj.(1,im.*y.(E))

     xx = range(sqrt(r0)-3.5,sqrt(r0)-1e-2, length=1000)
     xr = xx[convert.(Int64,argmin(abs.(f_in.(xx) .- real.(f_out.(xx)))))]
     E_q = (xr/R₀)^2*ħ^2/(2*mₑ1) + V1

     # Plot the criterion function
     plot!(p1,-xx,f_in.(xx),lw=2, ylims = (-15,15), labels =  false
        )
     plot!(p1,-xx,real.(f_out.(xx)),lw=2, ylims = (-15,15), labels =  false
        )
     plot!(p1,xlabel=L"Energy $E$", ylabel=L"$f(E)$", title="Criterion",
        )
     vline!(p1,[-r0], labels = false)
     scatter!(p1,[-xr],[f_in(xr)],mc=:red,ms=5,label=false)
     q  = sqrt.(-2*mₑ1*E_q/ħ^2)
     k₀ = sqrt.(2*mₑ1*(E_q.-V1)/ħ^2)


     # Define the integration ranges: r1 and r2 defines the inner and outer regions of the quantum dot
     r1 = range(0.00000000001, R₀, length=2000);
     r2 = range(R₀, 6.0*10^-9, length=2000);
     z₁ = r1.*k₀; z₂ = r2.*q;

     # Calculate the wavefunctions
     R₁ = sin.(z₁) ./ z₁.^2 .- cos.(z₁) ./ z₁; # Inner region
     R₂ = im.*exp.(.-z₂) .* (z₂ .+ 1) ./z₂.^2     ; # Outer region
    
     # Calculate the integrals
     a = integrate(r1, abs.(R₁.*r1).^2) # Inner region
     b = integrate(r2, abs.(R₂.*r2).^2) # Outer region

     # Calculate the normalization constant
     B_cnst = abs.(sin(k₀*R₀)/(k₀*R₀)^2 - cos(k₀*R₀)/(k₀*R₀))^2 / abs(exp(-q*R₀) * (q * R₀+1)/(q*R₀)^2)^2
    
     # Calculate the scaling factors
     A₀ = sqrt(16.0/(3*pi)) *1.0 / sqrt(a + b*B_cnst)
     B₀ = A₀*sqrt(B_cnst)
     plot!(p3, r1, (abs.(A₀*R₁).^2),
             lw=2,
             lc=clrs[1],
             label=false
         )
     plot!(p3, r2, (abs.(B₀*R₂).^2),
             lw=2,
             lc=clrs[1],
             label=false
         )
     plot!(p4, r1.*10^9, (A₀ * R₁ .* r1).^2*1e-9,
         lw=3,
         lc=clrs[1],
         label=false
     )
     plot!(p4, r2.*10^9,abs.(B₀ * R₂ .* r2).^2*1e-9,
         lw=3,
         lc=clrs[1],
         label=QD_label[1]
     )
     vline!(p3, (0, R₀), lc=clrs[1], lw=0.5, ls=:dashdot, label=false)
     vline!(p4, (0, R₀.*10^9), lc=clrs[1], lw=0.5, ls=:dashdot, label=false)
     tmp=(A₀ * R₁ .* r1)[end]^2
     scatter!(p4, (R₀.*10^9,tmp*1e-9),
            mc=clrs[1],
            yerr = [0.0],
            ma=1,
            ms = 7,
            # label=L"$R_{1}^2|\psi(R_{1})|^2$= %$(string(round(tmp/(10 ^(floor.(log.(10,tmp)))),digits=2)))×10\^{%$(convert(Int64,floor(log(10,tmp))))}"
            label=false; # L"$R_{1}^2|\psi(R_{1})|^2$"*lbls[1]
        )

    # Only core output data
    shell_dot[1,1] = R₀.*10^9
    shell_dot[1,2] = tmp*1e-9
    #Continuum line
    x_axis[:,1] = vcat(r1.*10^9,r2.*10^9)
    psi_y[:,1] = vcat((A₀ * R₁ .* r1).^2*1e-9,abs.(B₀ * R₂ .* r2).^2*1e-9)

    
    # # # calculation of the QD with the shell

     # Bound states criterion
     r0 = 2 * R₀^2 * (V2-V1) / ħ^2
     
     # Start a new local scope
     yy(E) = sqrt.(r0*mₑ2 .- E.^2.0 .*(mₑ2/mₑ1))

     ff_in(E) = E .* sphericalbesselj.(0,E)./ (mₑ1 .* sphericalbesselj.(1,E) ) .- 2/mₑ1 
     ff_out(E) = im .* yy.(E) .* sphericalhankelj.(0,im.*yy.(E))./ (mₑ2 .* sphericalhankelj.(1,im.*yy.(E)) ) .- 2/mₑ2
     
     xx  = range(sqrt(r0*mₑ1)-3.4,sqrt(r0*mₑ1)-1e-2, length=1000)
     xr₁ = xx[convert.(Int64,argmin(abs.(ff_in.(xx) .- real.(ff_out.(xx)))))]
     k₁  = xr₁/R₀
     q₂  = sqrt.(r0*mₑ2 .- xr₁^2*(mₑ2/mₑ1))/R₀
     xr₂ = q₂*R₀
     En  = (xr₁/R₀)^2*ħ^2/(2*mₑ1) + V1


     # Plot the criterion function
     plot!(p2,-xx,ff_in.(xx),lw=2, labels =  false
     )
     plot!(p2,-xx,real.(ff_out.(xx)),lw=2, labels =  false
     )
     plot!(p2,xlabel=L"Energy $E$", ylabel=L"$f(E)$", title="Criterion",
     )
     vline!(p2,[-r0*mₑ1], labels = false)
     scatter!(p2,[-xr₁],[ff_in(xr₁)],mc=:red,ms=5,label=false)
     xlabel!(p2, L"$k_1 R_0$");
     ylabel!(p2, L"$q_2 R_0$");
    #  display(plot(p1, p2, layout = grid(2,1)))
 

     q₃   = sqrt(-2 * mₑ1 * En / ħ^2)
     xr₂₃ = q₂ .* R_cs
     xr₃  = q₃ .* R_cs
     tmp = zeros(length(R_cs))
     for i in eachindex(R_cs)
         r1 = range(0.0000000000001, R₀, length=2000);
         r2 = range(R₀, R_cs[i], length=1000);
         r3 = range(R_cs[i], 6.0e-9, length=1000);
         z₁ = r1.*k₁; z₂ = im .* r2.*q₂; z₃ = im .* r3.*q₃;
 
         # Calculate the wavefunctions
         R₁ = sphericalbesselj.(1,z₁); # Inner region
         R₂ = im*imag.(sphericalhankelj.(1,z₂)); # Outer region
         R₃ = im*imag.(sphericalhankelj.(1,z₃)); # Outer region
         a  = integrate(r1, abs.(R₁.*r1).^2)
         b  = integrate(r2, abs.(R₂.*r2).^2)
         cc = integrate(r3, abs.(R₃.*r3).^2)
         B_cnst = -im.*sphericalbesselj.(1,xr₁) / imag.(sphericalhankelj.(1,im*xr₂))
         C_cnst = -im.*sphericalbesselj.(1,xr₁) * imag.(sphericalhankelj.(1,im*xr₂₃[i])) / (imag.(sphericalhankelj.(1,im*xr₃[i])) * imag.(sphericalhankelj.(1,im*xr₂))) 
 
         # #
         A₀ = sqrt(16.0/(3*pi)) *1.0 / sqrt(a + b*abs.(B_cnst)^2 + cc*abs.(C_cnst)^2);
         B₀ = im*imag(A₀*B_cnst);
         C₀ = im*imag(A₀*C_cnst);
 
         plot!(p3, r1, abs.(A₀*R₁).^2,
             lw=2,
             lc=clrs[i+1],
             label=false
         )
         plot!(p3, r2, abs.(B₀*R₂).^2,
             lw=2,
             lc=clrs[i+1],
             label=false
         )
         plot!(p3, r3, abs.(C₀*R₃).^2,
             lw=2,
             lc=clrs[i+1],
             label=false
         )
        #  println(log.((B₀*u₂.*r2).^2)[1]-log.((B₀*u₂.*r2).^2)[end])
         # #
 
         plot!(p4, r1.*10^9, abs.(A₀ * R₁ .* r1).^2*1e-9,
             lw=3,
             lc=clrs[i+1],
             label=QD_label[i+1]
         )
         plot!(p4, r2.*10^9,  abs.(B₀ * R₂ .* r2).^2*1e-9,
             lw=3,
             lc=clrs[i+1],
             label=false,
         )
         plot!(p4, r3.*10^9,  abs.(C₀ * R₃ .* r3).^2*1e-9,
             lw=3,
             lc=clrs[i+1],
             label=false, ytickfontsize=10,
         )
         vline!(p3, (0, R_cs[i]), lc=clrs[i+1], lw=0.5, ls=:dashdot, label=false)
         vline!(p4, (0, R_cs[i]*10^9), lc=clrs[i+1], lw=0.5, ls=:dashdot, label=false)
         tmp[i]=abs.(B₀ * R₂ .* r2)[end]^2

        x_axis[:,i+1] = vcat(r1.*10^9,r2.*10^9,r3.*10^9)
        psi_y[:,i+1]  = vcat((A₀ * R₁ .* r1).^2*1e-9,(B₀ * R₂ .* r2).^2*1e-9,(C₀ * R₃ .* r3).^2*1e-9)

    end
    for i in eachindex(R_cs)
        scatter!(p4, (R_cs[i].*10^9,tmp[i]*1e-9),
                mc=clrs[i+1],
                yerr = [0.0],
                ma=1,
                ms = 7,
                label=false; # L"$R_{%$(i)}^2|\psi(R_{%$(i)})|^2$"*lbls[i+1]
            ) 
        shell_dot[i+1,2] = tmp[i]*1e-9
        shell_dot[i+1,1] = R_cs[i].*10^9
    end

    plot!(p3, xlabel=L"$r$ (m)", ylabel=L"$|\psi(r)|$", title="Probability",
         legend=:topleft, legendcolumns=4
     )
    xlims!(p3, 0, 4e-9)
    plot!(p3, legendfontsize = 9, size = (800,600)) # Set the font size of the labels to 10 and the legend to 8
    # display(p3)
 
    plot!(p4, xlabel=L"$r$ (nm)", ylabel=L"$r^2|\psi(r)|^2$",
         legend=:topright, grid=false
     )
    xlims!(p4, 0, 4)
    ylims!(p4,0, 2.2)
    annotate!(p4, -0.21, 2.27, text("×10⁹", 11, :left))
    plot!(p4,legendfontsize = 12, size = (800,600), xguidefontsize=18, yguidefontsize=20,
    xtickfontsize=12, ytickfontsize = 12, grid = false,
    left_margin=4Plots.mm) # Set the font size of the labels to 10 and the legend to 8
    
    df = DataFrame(R_dis_1P = vec(x_axis), psi_dis_1P = vec(psi_y))

    # Write the DataFrame to a CSV file
    CSV.write("wavefunction_dis_1P.csv", df)
    println(shell_dot)
    return p4
end

function t_dis_1S_old(D00::Vector{Float64})
    # Constants
    ħ = 1.054e-34;  # Planck's constant
    m = 9.10938356e-31;  # Electron mass

    # Quantum dot size
    R₀ = 3.15e-9/2;  # Quantum dot sizes in meters
    R₁ = R₀ .+ 0.5.*D00.*10^-9; # Shell thickness
    # R₁ = R₀ .+ [0.64, 0.77, 1.75].*10^-9; # Shell thickness

    # Effective mass of electron
    mₑ1 = 0.13 * m;
    mₑ2 = 0.25 * m; 

    # Electron charge:
    q_e = 1.602176565e-19

    # Define initial potential energy
    V1 = cond_bnd_edge((2*R₀)*10^9)*q_e ;
    # V1 = -4.0*q_e;
    V2 = V1 + 1.4*q_e;

    t_coupling = zeros(length(R₁)+1);


    # # Calculation for the Bare QD, i.e., no Shell

    r0 = -2*mₑ1*V1*R₀^2/ħ^2

    # Start a new local scope
    xx = range(2,3, length=1000)
    x_a = xx[convert.(Int64,argmin(abs.(map(x1->x1^2 + x1^2*(cot(x1))^2-r0,xx))))]
    y_a = -x_a*cot(x_a)
    q = y_a./R₀;
    k₀ = x_a./R₀;

    r1 = range(0.000000000001,R₀, length=1000);
    r2 = range(R₀,6.0*10^-9, length=1000);

    u₁ = sin.(k₀ * r1) ./ r1;
    u₂ = exp.(-q * r2) ./ r2;

    a = integrate(r1, abs.(u₁.*r1).^2)
    b = integrate(r2, abs.(u₂.*r2).^2)

    A_cnst = exp(-2*q*R₀) / (sin(k₀ * R₀)^2)
    
    B₀ = 1.0 / sqrt(a*A_cnst + b)
    t_coupling[1] = B₀*integrate(r2, u₂.*(r2).^2)


    # # Calculation for the CdSe/ZnS core/shell QD


    # Bound states criterion
    r0 = 2 * R₀^2 * (V2-V1) / ħ^2

    xx = range(1.5, 2.5, length=5000)
    x_a= xx[convert.(Int64, argmin(abs.(map(x1 -> x1^2/ mₑ1 + mₑ2 * (x1 * cot(x1)/ mₑ1 + 1/mₑ2 - 1/mₑ1 )^2 - r0, xx))))]
    #  y_a= -mₑ2 * x_a * cot(x_a)/mₑ1 + mₑ2/mₑ1 - 1
    y_a= sqrt(mₑ2* (r0-x_a^2/mₑ1 ) )

    q₂ = y_a / R₀;
    k₁ = x_a / R₀


    En = ħ^2 * k₁^2 / (2 * mₑ1) + V1
    q₃ = sqrt(-2 * mₑ1 * En / ħ^2)
    for i in 1:length(R₁)
        r1 = range(0.000000000001, R₀, length=2000);
        r2 = range(R₀, R₁[i]  , length=1000);
        r3 = range(R₁[i], 6.0e-9, length=1000);

        u₁ = sin.(k₁ * r1) ./ r1;
        u₂ = exp.(-q₂ * r2) ./ r2;
        u₃ = exp.(-q₃ * r3) ./ r3;
        a = integrate(r1, abs.(u₁.*r1).^2)
        b = integrate(r2, abs.(u₂.*r2).^2)
        cc = integrate(r3, abs.(u₃.*r3).^2)
        A_cnst = exp(-2*q₂*R₀) / (sin(k₁ * R₀)^2)
        C_cnst = exp(2*R₁[i]*(q₃ - q₂))

        # #
        B₀ = 1.0 / sqrt(a*A_cnst + b + cc*C_cnst);
        C₀ = B₀*sqrt(C_cnst);
        t_coupling[i+1] = C₀*integrate(r3,u₃.*r3.^2)
    end

    return t_coupling
end

function t_dis_1P_old(D00::Vector{Float64})
    # Constants
    ħ = 1.054e-34;  # Planck's constant
    m = 9.10938356e-31;  # Electron mass
 
    # Quantum dot size
    R₀   = 3.154e-9/2;  # Quantum dot sizes in meters
    R_cs = R₀ .+ 0.5.*D00.*10^-9; # Shell thickness
    # R₀ = 2.74e-9/2;  
    # R₁ = R₀ + 1e-9;
    # Effective mass of electron
    mₑ1 = 0.13 * m;
    mₑ2 = 0.25 * m; 
 
    # Electron charge:
    q_e = 1.602176565e-19
 
    # Define initial potential energy
    V1 = cond_bnd_edge((2*R₀)*10^9)*q_e ;
    V2 = V1 + 1.4*q_e;

    # Definition of the spherical Bessel and Hankel functions
    sphericalhankelj(ν, x) = √(π/2x).*hankelh1(ν+1/2, x)
    sphericalbesselj(ν, x) = √(π/2x).*besselj(ν+1/2, x)

    t_coupling = zeros(length(R_cs)+1);
 
    # # Calculation for the Bare QD, i.e., no Shell
     r0 = -2*mₑ1*V1*R₀^2/ħ^2     

     # Start a new local scope
     y(x) = sqrt.(r0 .- x.^2)

     f_in(x) = x .* sphericalbesselj.(0,x)./ sphericalbesselj.(1,x) 
     f_out(x) = im .* y.(x) .* sphericalhankelj.(0,im.*y.(x))./ sphericalhankelj.(1,im.*y.(x))

     xx = range(sqrt(r0)-3.5,sqrt(r0)-1e-2, length=1000)
     xr = xx[convert.(Int64,argmin(abs.(f_in.(xx) .- real.(f_out.(xx)))))]
     E_q = (xr/R₀)^2*ħ^2/(2*mₑ1) + V1

     q  = sqrt.(-2*mₑ1*E_q/ħ^2)
     k₀ = sqrt.(2*mₑ1*(E_q.-V1)/ħ^2)


     # Define the integration ranges: r1 and r2 defines the inner and outer regions of the quantum dot
     r1 = range(0.00000000001, R₀, length=1000);
     r2 = range(R₀, 6.0*10^-9, length=1000);
     z₁ = r1.*k₀; z₂ = r2.*q;

     # Calculate the wavefunctions
     R₁ = sin.(z₁) ./ z₁.^2 .- cos.(z₁) ./ z₁; # Inner region
     R₂ = im.*exp.(.-z₂) .* (z₂ .+ 1) ./z₂.^2     ; # Outer region
    
     # Calculate the integrals
     a = integrate(r1, abs.(R₁.*r1).^2) # Inner region
     b = integrate(r2, abs.(R₂.*r2).^2) # Outer region

     # Calculate the normalization constant
     B_cnst = abs.(sin(k₀*R₀)/(k₀*R₀)^2 - cos(k₀*R₀)/(k₀*R₀))^2 / abs(exp(-q*R₀) * (q * R₀+1)/(q*R₀)^2)^2
    
     # Calculate the scaling factors
     A₀ = sqrt(16.0/(3*pi)) *1.0 / sqrt(a + b*B_cnst)
     B₀ = -im*A₀*sqrt(B_cnst)
    
     #Electronic coupling of the bare QD
     t_coupling[1] = B₀*integrate(r2, R₂.*(r2).^2)

    # # # calculation of the QD with the shell

     # Bound states criterion
     r0 = 2 * R₀^2 * (V2-V1) / ħ^2
     
     # Start a new local scope
     yy(E) = sqrt.(r0*mₑ2 .- E.^2.0 .*(mₑ2/mₑ1))

     ff_in(E) = E .* sphericalbesselj.(0,E)./ (mₑ1 .* sphericalbesselj.(1,E) ) .- 2/mₑ1 
     ff_out(E) = im .* yy.(E) .* sphericalhankelj.(0,im.*yy.(E))./ (mₑ2 .* sphericalhankelj.(1,im.*yy.(E)) ) .- 2/mₑ2
     
     xx  = range(sqrt(r0*mₑ1)-3.4,sqrt(r0*mₑ1)-1e-2, length=1000)
     xr₁ = xx[convert.(Int64,argmin(abs.(ff_in.(xx) .- real.(ff_out.(xx)))))]
     k₁  = xr₁/R₀
     q₂  = sqrt.(r0*mₑ2 .- xr₁^2*(mₑ2/mₑ1))/R₀
     xr₂ = q₂*R₀
     En  = (xr₁/R₀)^2*ħ^2/(2*mₑ1) + V1



     q₃   = sqrt(-2 * mₑ1 * En / ħ^2)
     xr₂₃ = q₂ .* R_cs
     xr₃  = q₃ .* R_cs
     for i in eachindex(R_cs)
         r1 = range(0.0000000000001, R₀, length=2000);
         r2 = range(R₀, R_cs[i], length=1000);
         r3 = range(R_cs[i], 6.0e-9, length=1000);
         z₁ = r1.*k₁; z₂ = im .* r2.*q₂; z₃ = im .* r3.*q₃;
 
         # Calculate the wavefunctions
         R₁ = sphericalbesselj.(1,z₁); # Inner region
         R₂ = im*imag.(sphericalhankelj.(1,z₂)); # Outer region
         R₃ = im*imag.(sphericalhankelj.(1,z₃)); # Outer region
         a  = integrate(r1, abs.(R₁.*r1).^2)
         b  = integrate(r2, abs.(R₂.*r2).^2)
         cc = integrate(r3, abs.(R₃.*r3).^2)
         B_cnst = -im.*sphericalbesselj.(1,xr₁) / imag.(sphericalhankelj.(1,im*xr₂))
         C_cnst = -im.*sphericalbesselj.(1,xr₁) * imag.(sphericalhankelj.(1,im*xr₂₃[i])) / (imag.(sphericalhankelj.(1,im*xr₃[i])) * imag.(sphericalhankelj.(1,im*xr₂))) 
 
         # #
         A₀ = sqrt(16.0/(3*pi)) *1.0 / sqrt(a + b*abs.(B_cnst)^2 + cc*abs.(C_cnst)^2);
         B₀ = im*imag(A₀*B_cnst);
         C₀ = im*imag(A₀*C_cnst);
        
         t_coupling[i+1] = C₀*integrate(r3,R₃.*r3.^2)
    end

    return t_coupling
end

function layout_plot()
    l = @layout [a b ; c d]
    p1 = ψ_size_plot_1S()
    p2 = ψ_size_plot_1P()
    p3 = ψ_dis_plot_1S()
    p4 = ψ_dis_plot_1P()
    return plot(p1, p2, p3, p4, layout = l) # savefig(plot(p1, p2, p3, p4, layout = l),"/media/r_floren/FernandoM/Research/Tim_Liam/Theory_paper/Figures/wavefunctions")
end

 # tmp = ψ_size_plot_1P()
# savefig(tmp,"tmp.png")

function tmp()
    function cond_bnd_edge(dis::Float64)
        return -3.49 + 2.97*dis^(-1.24)
    end
    D00 = [2.6, 3.0, 3.8, 5.6] # Quantum dot sizes (diameter) in nanometers
    # Constants
    ħ = 1.054e-34;  # Planck's constant
    m = 9.10938356e-31;  # Electron mass
    converter = 6.241495961752113e18;  # eV to J conversion factor

    # Quantum dot size
    R₀ = D00 .* 1e-9 ./2 # # Quantum dot sizes (diameter) in nanometers
    # Effective mass of electron
    mₑ = 0.13 * m;

    # Electron charge:
    q_e = 1.602176565e-19

    # Define initial potential energy
    V0 = cond_bnd_edge.(D00).*q_e ;



    f(x, y) = y^2*x*cos(x) - (x^2*(1+y)+y^2)*sin(x)

    x = range(0, 10, length=100)
    y = range(0, 10, length=100)
    z = @. f(x', y)
    p1 = plot()
    contour!(p1, x, y, z, levels=[0.0], color=:turbo, clabels=false, cbar=false, lw=1, xlims=(0,10), ylims=(0,10))
    title!(p1, L"Plot of $(3x + y^2)|\sin(x) + \cos(y)|$")
    xlabel!(p1, L"x")
    ylabel!(p1, L"y")

    # Bound state criterion
    r0 =  -2*mₑ*V0 .* R₀ .^2 ./ ħ^2
    for i in r0
        g(x,y) = x^2 + y^2 - i
        z2 = @. g(x', y)
        contour!(p1, x, y, z2, levels=[0.0], color=:black, lw=2, label="Bound State Criterion", aspect_ratio=:equal)
    end
    
    return p1
end


function tmp2()
    function cond_bnd_edge(dis::Float64)
        return -3.49 + 2.97*dis^(-1.24)
    end
    D00 = range(2.6, 6.0, length=100) # Quantum dot sizes (diameter) in nanometers
    # Constants
    ħ = 1.054e-34;  # Planck's constant
    m = 9.10938356e-31;  # Electron mass
    converter = 6.241495961752113e18;  # eV to J conversion factor

    # Quantum dot size
    R₀ = D00 .* 1e-9 ./2 # # Quantum dot sizes (diameter) in nanometers
    # Effective mass of electron
    mₑ = 0.13 * m;

    # Electron charge:
    q_e = 1.602176565e-19

    # Define initial potential energy
    V0 = cond_bnd_edge.(D00).*q_e ;

    # Bound state criterion
    r0 =  -2*mₑ*V0 .* R₀ .^2 ./ ħ^2

    x1 = range(1, 5, length=1000)
    

    x2 = range(5, 8, length=1000)


    ξ1 = zeros(Float64, length(D00))
    ξ2 = zeros(Float64, length(D00))

    η1 = zeros(Float64, length(D00))
    η2 = zeros(Float64, length(D00))

    for i in eachindex(D00)
        if abs(sqrt(r0[i])) < 2*pi
            continue
        end
        println(i)
        y1 = sqrt.(abs.(r0[i] .- x1.^2))
        y2 = sqrt.(abs.(r0[i] .- x2.^2))
        ξ1[i] = x1[findmin(abs.( y1.^2 .*x1.*cos.(x1) .- (x1.^2 .*(1.0 .+y1).+y1.^2).*sin.(x1) ))[2]]
        η1[i] = sqrt.(r0[i] .- ξ1[i]^2)
        ξ2[i] = x2[findmin(abs.( y2.^2 .*x2.*cos.(x2) .- (x2.^2 .*(1.0 .+y2).+y2.^2).*sin.(x2) ))[2]]
        η2[i] = sqrt.(r0[i] .- ξ2[i]^2)
    end
    
    p1 = plot()
    # plot!(p1, D00,-converter .* ħ^2 ./(2 .* mₑ .* R₀.^2) .* η1.^2)
    plot!(p1, D00, converter .*ħ^2 ./(2 .* mₑ .* R₀.^2) .* ξ1.^2 .+ converter .* V0, label="First Bound State")

    # plot!(p2, D00,-converter .* ħ^2 ./(2 .* mₑ .* R₀.^2) .* η2.^2)
    plot!(p1, D00, converter .*ħ^2 ./(2 .* mₑ .* R₀.^2) .* ξ2.^2 .+ converter .* V0, label="Second Bound State")

    display(p1)
end
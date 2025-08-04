function permanovas(dm::Matrix, metadatums::Vector{<:AbstractVector}; n = 1000, mdlabels = ["md$i" for i in eachindex(metadatums)])
    permdf = mapreduce(vcat, zip(metadatums, mdlabels)) do (md, lab)    
        hasmd = findall(!ismissing, md)
        df = DataFrame(test = md[hasmd])
        disallowmissing!(df)
        size(df, 1) < 20 && @warn "Very small number of not missing $lab"

        p = permanova(df, dm[hasmd, hasmd], @formula(1 ~ test), n)

        return (; metadatum = lab, varexpl=varexpl(p), pvalue=pvalue(p))
    end

    return DataFrame(permdf)
end

function permanovas(dms::AbstractArray{<:AbstractMatrix}, metadatums; n = 1000, commlabels = [], mdlabels = ["md$i" for i in eachindex(metadatums)])
    isempty(commlabels) && (commlabels = ["comm$i" for i in eachindex(dms)])
    
    mapreduce(vcat, enumerate(dms)) do (i, dm)
        @info "Permanovas for $(commlabels[i])"
        df = permanovas(dm, metadatums; n, mdlabels)
        df.label .= commlabels[i]
        return df
    end
end

varexpl(p::PERMANOVA.PSummary) = p.results[1, 3] * 100
pvalue(p::PERMANOVA.PSummary) = p.results[1, 5]

function plot_permanovas(pdf; commlabels=[], mdlabels=[], colormap=:blues, colorrange=(0,10))
    fig = Figure()
    ax = Axis(fig[1,1])
    hm = plot_permanovas!(ax, pdf; commlabels, mdlabels, colormap, colorrange)

    return fig, ax, hm
end

function plot_permanovas!(ax, pdf; commlabels=unique(pdf.label), mdlabels=unique(pdf.metadatum), colormap=:blues, colorrange=(0,10))
    pgrp = groupby(pdf, :label)
    
    vmat = mapreduce(df-> df.varexpl, hcat, pgrp)
    pmat = mapreduce(df-> df.pvalue, hcat, pgrp)
    
    hm = CairoMakie.heatmap!(ax, vmat'; colormap, colorrange)

    for ci in CartesianIndices(pmat)
        c = vmat[ci] < (colorrange[2] - colorrange[1]) / 2 ? :black : :lightgray
        text!(ax, string(round(vmat[ci], digits=2), "%"); position=(ci[2],ci[1]), align=(:center, :center), color=c)
        p = pmat[ci]
        stars = p < 0.001 ? "***" : p < 0.01 ? "**" : p < 0.05 ? "*" : ""
        text!(ax, stars; position=(ci[2],ci[1]), align=(:center, :bottom), color=c)
    end

    ax.xticks = (1:length(commlabels), commlabels)
    ax.yticks = (1:length(mdlabels), mdlabels)

    return hm
end
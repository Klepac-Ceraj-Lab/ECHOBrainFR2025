#####
# Model 1 - brain, age, sex
#####
using Chain
using CSV
using Statistics
using Random
using DataFrames
using CategoricalArrays
using CairoMakie
using GLMNet
using MLJ
using GLM
using DecisionTree              # Needed to extract feature importances from the classical forest
using XGBoost                   # Needed to extract feature importances from the booster
using MLJDecisionTreeInterface  # Provides the RandomForesRegressor for MLJ
using MLJXGBoostInterface       # Provides the XGBoostRegressor for MLJ
using Distances
using MultivariateStats
using LinearAlgebra
using PERMANOVA
using ECHOBrainFR2025

## Loading Data
input_mdata, input_brain_raw, _, _ = load_brain_data()
include("notebooks/90-preprocess_brain_data.jl")
sort!(final_inputs, :UID)
sort!(final_inputs_rclr, :UID)

bin1_inpts_tivnorm = subset(final_inputs, :agebin => x -> x .== 1)
bin2_inpts_tivnorm = subset(final_inputs, :agebin => x -> x .== 2)
bin3_inpts_tivnorm = subset(final_inputs, :agebin => x -> x .== 3)
bin1_inpts_rclr = subset(final_inputs_rclr, :agebin => x -> x .== 1)
bin2_inpts_rclr = subset(final_inputs_rclr, :agebin => x -> x .== 2)
bin3_inpts_rclr = subset(final_inputs_rclr, :agebin => x -> x .== 3)

#####
# Creating Figure 1
#####

fig = Figure( size = (1600, 1600))

ABC_Subfig = GridLayout(fig[1,:]; alignmomde = Outside())
A_Subfig = GridLayout(ABC_Subfig[:,1]; alignmomde = Outside())
B_Subfig = GridLayout(ABC_Subfig[:,2]; alignmomde = Outside())
C_Subfig = GridLayout(ABC_Subfig[:,3]; alignmomde = Outside())
DEF_Subfig = GridLayout(fig[2,:]; alignmomde = Outside())
D_Subfig = GridLayout(DEF_Subfig[:,1]; alignmomde = Outside())
E_Subfig = GridLayout(DEF_Subfig[:,2]; alignmomde = Outside())
F_Subfig = GridLayout(DEF_Subfig[:,3]; alignmomde = Outside())
GHI_Subfig = GridLayout(fig[2,:]; alignmomde = Outside())
G_Subfig = GridLayout(GHI_Subfig[:,1]; alignmomde = Outside())
H_Subfig = GridLayout(GHI_Subfig[:,2]; alignmomde = Outside())
I_Subfig = GridLayout(GHI_Subfig[:,3]; alignmomde = Outside())

age_bins = collect(1:1:15)
age_cmap = :cividis
age_cgrad = cgrad(age_cmap, age_bins, categorical=false, rev=true)
age_cgrad_discrete = cgrad(age_cmap, (length(age_bins)-1), categorical=true, rev=true)

bmi_bins = collect(-2.5:0.5:5.0)
bmi_cmap = :plasma
bmi_cgrad = cgrad(bmi_cmap, bmi_bins, categorical=false, rev=true)
bmi_cgrad_discrete = cgrad(bmi_cmap, (length(bmi_bins)-1), categorical=true, rev=true)

fr_bins = collect(1.0:0.5:5.5)
fr_cmap = :viridis
fr_cgrad = cgrad(fr_cmap, fr_bins, categorical=false, rev=true)
fr_cgrad_discrete = cgrad(fr_cmap, (length(fr_bins)-1), categorical=true, rev=true)

sex_bins = [1, 2]
sex_cmap = :seismic  # 1 is red, 2 is blue
sex_cgrad = cgrad(sex_cmap, 2, categorical=true, rev=true)

edu_bins = [1, 2]
edu_cmap = :fruitpunch
edu_cgrad = cgrad(edu_cmap, 2, categorical=true, rev=true)

#####
# Panel A - Scatterplot for Bin 1
#####

AxATopHist = Axis(
    A_Subfig[1,1],
    xticks = 0.5:0.5:5.0,
    xlabel = "Food Responsiveness score",
    ylabel = "Absolute frequency")
AxAScatter = Axis(
    A_Subfig[2,1],
    xticks = 0.5:0.5:5.0,
    yticks = -2.5:0.5:4.5,
    xlabel = "Food Responsiveness score",
    ylabel = "WHO zBMI")
AxARightHist = Axis(
    A_Subfig[2,2],
    yticks = -2.5:0.5:4.5,
    xlabel = "Absolute frequency",
    ylabel = "WHO zBMI"
)

xlims!(AxATopHist, [0.49, 5.01])
tightlimits!(AxATopHist, Bottom())
xlims!(AxAScatter, [0.49, 5.01])
ylims!(AxAScatter, [-2.51, +4.51])
ylims!(AxARightHist, [-2.51, +4.51])
tightlimits!(AxARightHist, Left())

linkxaxes!(AxATopHist, AxAScatter)
linkyaxes!(AxARightHist, AxAScatter)
hidexdecorations!(AxATopHist)
hideydecorations!(AxATopHist, ticks = false)
hidexdecorations!(AxARightHist, ticks = false)
hideydecorations!(AxARightHist)
hidedecorations!(AxAScatter, label = false, ticks = false, ticklabels = false)

sc = scatter!(
    AxAScatter,
    bin1_inpts_rclr.fr_score,
    bin1_inpts_rclr.zBMI,
    color = bin1_inpts_rclr.sex,
    colormap = :seismic,
    marker = [ ( (el == 1) ? (:rect) : (:utriangle) ) for el in bin1_inpts_rclr.mom_edu_cat ], # 1 is circle, 2 is rectangle
    markersize = [ log2(2*el)^2 for el in bin1_inpts_rclr.fr_score_age_years ]
)

r_bmi_fr = round(cor(bin1_inpts_rclr.fr_score, bin1_inpts_rclr.zBMI), digits = 3)
r_bmi_p = round(pearson_pvalue(r_bmi_fr, nrow(bin1_inpts_rclr)), digits = 6)

annotations!(
    AxAScatter,
    [ "R = $(r_bmi_fr), p = $(r_bmi_p)" ],
    [ Point(2.75, -2.5) ],
    color = :black
)

hstp = hist!(
    AxATopHist,
    bin1_inpts_rclr.fr_score,
    bins = fr_bins,
    color = collect(fr_cgrad_discrete),
    strokecolor = :black,
    strokewidth = 1
)

hsri = hist!(
    AxARightHist,
    bin1_inpts_rclr.zBMI,
    color = collect(bmi_cgrad_discrete),
    bins = bmi_bins,
    strokecolor = :black,
    strokewidth = 1,
    direction=:x
)

Legend(A_Subfig[1,2],
    [
        [
            MarkerElement(marker = :circle, color = sex_cgrad[1]),
            MarkerElement(marker = :circle, color = sex_cgrad[end])
        ],
        [
            MarkerElement(marker = :rect, color = :black),
            MarkerElement(marker = :utriangle, color = :black)
        ],
        [
            MarkerElement(marker = :circle, markersize = log2(2*2.0)^2, color = age_cgrad[1], strokewidth = 1, strokecolor = :black),
            MarkerElement(marker = :circle, markersize = log2(2*8.0)^2,color = age_cgrad[128], strokewidth = 1, strokecolor = :black),
            MarkerElement(marker = :circle, markersize = log2(2*15.0)^2,color = age_cgrad[end], strokewidth = 1, strokecolor = :black)
        ],
    ],
    [
        [
            "Female",
            "Male"
        ],
        [
            "No",
            "Yes"
        ],
        [
            "2",
            "8",
            "15"
        ],
        
    ],
    [
        "Sex",
        "Education (uni/college)",
        "Age"
    ];
    groupgap = 5,
    rowgap = 2, 
    colgap = 5,
    titlegap = 1, 
    tellheight = false, 
    tellwidth = false, 
    nbanks = 3
)

rowsize!(A_Subfig, 1, Relative(0.3))
colsize!(A_Subfig, 2, Relative(0.3))

#####
# Panel B - Scatterplot for Bin 2
#####

AxBTopHist = Axis(
    B_Subfig[1,1],
    xticks = 0.5:0.5:5.0,
    xlabel = "Food Responsiveness score",
    ylabel = "Absolute frequency")
AxBScatter = Axis(
    B_Subfig[2,1],
    xticks = 0.5:0.5:5.0,
    yticks = -2.5:0.5:4.5,
    xlabel = "Food Responsiveness score",
    ylabel = "WHO zBMI")
AxBRightHist = Axis(
    B_Subfig[2,2],
    yticks = -2.5:0.5:4.5,
    xlabel = "Absolute frequency",
    ylabel = "WHO zBMI"
)

xlims!(AxBTopHist, [0.49, 5.01])
tightlimits!(AxBTopHist, Bottom())
xlims!(AxBScatter, [0.49, 5.01])
ylims!(AxBScatter, [-2.51, +4.51])
ylims!(AxBRightHist, [-2.51, +4.51])
tightlimits!(AxBRightHist, Left())

linkxaxes!(AxBTopHist, AxBScatter)
linkyaxes!(AxBRightHist, AxBScatter)
hidexdecorations!(AxBTopHist)
hideydecorations!(AxBTopHist, ticks = false)
hidexdecorations!(AxBRightHist, ticks = false)
hideydecorations!(AxBRightHist)
hidedecorations!(AxBScatter, label = false, ticks = false, ticklabels = false)

sc = scatter!(
    AxBScatter,
    bin2_inpts_rclr.fr_score,
    bin2_inpts_rclr.zBMI,
    color = bin2_inpts_rclr.sex,
    colormap = :seismic,
    marker = [ ( (el == 1) ? (:rect) : (:utriangle) ) for el in bin2_inpts_rclr.mom_edu_cat ], # 1 is circle, 2 is rectangle
    markersize = [ log2(2*el)^2 for el in bin2_inpts_rclr.fr_score_age_years ]
)

r_bmi_fr = round(cor(bin2_inpts_rclr.fr_score, bin2_inpts_rclr.zBMI), digits = 3)
r_bmi_p = round(pearson_pvalue(r_bmi_fr, nrow(bin2_inpts_rclr)), digits = 6)

annotations!(
    AxBScatter,
    [ "R = $(r_bmi_fr), p = $(r_bmi_p)" ],
    [ Point(2.75, -2.5) ],
    color = :black
)

hstp = hist!(
    AxBTopHist,
    bin2_inpts_rclr.fr_score,
    bins = fr_bins,
    color = collect(fr_cgrad_discrete),
    strokecolor = :black,
    strokewidth = 1
)

hsri = hist!(
    AxBRightHist,
    bin2_inpts_rclr.zBMI,
    color = collect(bmi_cgrad_discrete),
    bins = bmi_bins,
    strokecolor = :black,
    strokewidth = 1,
    direction=:x
)

Legend(B_Subfig[1,2],
    [
        [
            MarkerElement(marker = :circle, color = sex_cgrad[1]),
            MarkerElement(marker = :circle, color = sex_cgrad[end])
        ],
        [
            MarkerElement(marker = :rect, color = :black),
            MarkerElement(marker = :utriangle, color = :black)
        ],
        [
            MarkerElement(marker = :circle, markersize = log2(2*2.0)^2, color = age_cgrad[1], strokewidth = 1, strokecolor = :black),
            MarkerElement(marker = :circle, markersize = log2(2*8.0)^2,color = age_cgrad[128], strokewidth = 1, strokecolor = :black),
            MarkerElement(marker = :circle, markersize = log2(2*15.0)^2,color = age_cgrad[end], strokewidth = 1, strokecolor = :black)
        ],
    ],
    [
        [
            "Female",
            "Male"
        ],
        [
            "No",
            "Yes"
        ],
        [
            "2",
            "8",
            "15"
        ],
        
    ],
    [
        "Sex",
        "Education (uni/college)",
        "Age"
    ];
    groupgap = 5,
    rowgap = 2, 
    colgap = 5,
    titlegap = 1, 
    tellheight = false, 
    tellwidth = false, 
    nbanks = 3
)

rowsize!(B_Subfig, 1, Relative(0.3))
colsize!(B_Subfig, 2, Relative(0.3))

#####
# Panel C - Scatterplot for Bin 3
#####

AxCTopHist = Axis(
    C_Subfig[1,1],
    xticks = 0.5:0.5:5.0,
    xlabel = "Food Responsiveness score",
    ylabel = "Absolute frequency")
AxCScatter = Axis(
    C_Subfig[2,1],
    xticks = 0.5:0.5:5.0,
    yticks = -2.5:0.5:4.5,
    xlabel = "Food Responsiveness score",
    ylabel = "WHO zBMI")
AxCRightHist = Axis(
    C_Subfig[2,2],
    yticks = -2.5:0.5:4.5,
    xlabel = "Absolute frequency",
    ylabel = "WHO zBMI"
)

xlims!(AxCTopHist, [0.49, 5.01])
tightlimits!(AxCTopHist, Bottom())
xlims!(AxCScatter, [0.49, 5.01])
ylims!(AxCScatter, [-2.51, +4.51])
ylims!(AxCRightHist, [-2.51, +4.51])
tightlimits!(AxCRightHist, Left())

linkxaxes!(AxCTopHist, AxCScatter)
linkyaxes!(AxCRightHist, AxCScatter)
hidexdecorations!(AxCTopHist)
hideydecorations!(AxCTopHist, ticks = false)
hidexdecorations!(AxCRightHist, ticks = false)
hideydecorations!(AxCRightHist)
hidedecorations!(AxCScatter, label = false, ticks = false, ticklabels = false)

sc = scatter!(
    AxCScatter,
    bin3_inpts_rclr.fr_score,
    bin3_inpts_rclr.zBMI,
    color = bin3_inpts_rclr.sex,
    colormap = :seismic,
    marker = [ ( (el == 1) ? (:rect) : (:utriangle) ) for el in bin3_inpts_rclr.mom_edu_cat ], # 1 is circle, 2 is rectangle
    markersize = [ log2(2*el)^2 for el in bin3_inpts_rclr.fr_score_age_years ]
)

r_bmi_fr = round(cor(bin3_inpts_rclr.fr_score, bin3_inpts_rclr.zBMI), digits = 3)
r_bmi_p = round(pearson_pvalue(r_bmi_fr, nrow(bin3_inpts_rclr)), digits = 6)

annotations!(
    AxCScatter,
    [ "R = $(r_bmi_fr), p = $(r_bmi_p)" ],
    [ Point(2.75, -2.5) ],
    color = :black
)

hstp = hist!(
    AxCTopHist,
    bin3_inpts_rclr.fr_score,
    bins = fr_bins,
    color = collect(fr_cgrad_discrete),
    strokecolor = :black,
    strokewidth = 1
)

hsri = hist!(
    AxCRightHist,
    bin3_inpts_rclr.zBMI,
    color = collect(bmi_cgrad_discrete),
    bins = bmi_bins,
    strokecolor = :black,
    strokewidth = 1,
    direction=:x
)

Legend(C_Subfig[1,2],
    [
        [
            MarkerElement(marker = :circle, color = sex_cgrad[1]),
            MarkerElement(marker = :circle, color = sex_cgrad[end])
        ],
        [
            MarkerElement(marker = :rect, color = :black),
            MarkerElement(marker = :utriangle, color = :black)
        ],
        [
            MarkerElement(marker = :circle, markersize = log2(2*2.0)^2, color = age_cgrad[1], strokewidth = 1, strokecolor = :black),
            MarkerElement(marker = :circle, markersize = log2(2*8.0)^2,color = age_cgrad[128], strokewidth = 1, strokecolor = :black),
            MarkerElement(marker = :circle, markersize = log2(2*15.0)^2,color = age_cgrad[end], strokewidth = 1, strokecolor = :black)
        ],
    ],
    [
        [
            "Female",
            "Male"
        ],
        [
            "No",
            "Yes"
        ],
        [
            "2",
            "8",
            "15"
        ],
        
    ],
    [
        "Sex",
        "Education (uni/college)",
        "Age"
    ];
    groupgap = 5,
    rowgap = 2, 
    colgap = 5,
    titlegap = 1, 
    tellheight = false, 
    tellwidth = false, 
    nbanks = 3
)

rowsize!(C_Subfig, 1, Relative(0.3))
colsize!(C_Subfig, 2, Relative(0.3))

#####
# Panel D - Correlogram for Bin 1
#####

cor_vars = [:fr_score, :zBMI, :fr_score_age_years, :sex, :mom_edu_cat]
cor_data = select(bin1_inpts_rclr, cor_vars)

# Compute correlation matrix
corr_matrix = pairwise(cor, eachcol(cor_data))
# corr_matrix = cor(Matrix(data))

for i in 1:size(corr_matrix, 1)
    for j in i:size(corr_matrix, 2)
        corr_matrix[i, j] = NaN
    end
end

corr_matrix = corr_matrix[2:end, 1:end-1]

# Plot correlogram
AxDCorr = Axis(
    D_Subfig[1,1],
    xticks = (1:4, ["FR score", "zBMI", "age", "sex"]),
    yticks = (1:4, ["zBMI", "age", "sex", "Maternal\nEducation"]),
    aspect = 1,
    yreversed = true
)

corr_bins = collect(-1.0:0.1:1.0)
corr_cmap = :coolwarm
corr_cgrad = cgrad(corr_cmap, corr_bins, categorical=false, rev=false)

hm_corr = heatmap!(
    AxDCorr,
    corr_matrix',
    colormap = corr_cgrad,  # diverging color map
    colorrange = (-1, 1)
)

cbar_fr = Colorbar( D_Subfig[1,2], hm_corr, vertical = true)

for ci in CartesianIndices(corr_matrix)
    !isnan(corr_matrix[ci]) && text!(AxDCorr, string(round(corr_matrix[ci], digits=3)); position=(ci[2],ci[1]), align=(:center, :center))
end

#####
# Panel E - Correlogram for Bin 2
#####

cor_vars = [:fr_score, :zBMI, :fr_score_age_years, :sex, :mom_edu_cat]
cor_data = select(bin2_inpts_rclr, cor_vars)

# Compute correlation matrix
corr_matrix = pairwise(cor, eachcol(cor_data))
# corr_matrix = cor(Matrix(data))

for i in 1:size(corr_matrix, 1)
    for j in i:size(corr_matrix, 2)
        corr_matrix[i, j] = NaN
    end
end

corr_matrix = corr_matrix[2:end, 1:end-1]

# Plot correlogram
AxECorr = Axis(
    E_Subfig[1,1],
    xticks = (1:4, ["FR score", "zBMI", "age", "sex"]),
    yticks = (1:4, ["zBMI", "age", "sex", "Maternal\nEducation"]),
    aspect = 1,
    yreversed = true
)

corr_bins = collect(-1.0:0.1:1.0)
corr_cmap = :coolwarm
corr_cgrad = cgrad(corr_cmap, corr_bins, categorical=false, rev=false)

hm_corr = heatmap!(
    AxECorr,
    corr_matrix',
    colormap = corr_cgrad,  # diverging color map
    colorrange = (-1, 1)
)

cbar_fr = Colorbar( E_Subfig[1,2], hm_corr, vertical = true)

for ci in CartesianIndices(corr_matrix)
    !isnan(corr_matrix[ci]) && text!(AxECorr, string(round(corr_matrix[ci], digits=3)); position=(ci[2],ci[1]), align=(:center, :center))
end

#####
# Panel F - Correlogram for Bin 3
#####

cor_vars = [:fr_score, :zBMI, :fr_score_age_years, :sex, :mom_edu_cat]
cor_data = select(bin3_inpts_rclr, cor_vars)

# Compute correlation matrix
corr_matrix = pairwise(cor, eachcol(cor_data))
# corr_matrix = cor(Matrix(data))

for i in 1:size(corr_matrix, 1)
    for j in i:size(corr_matrix, 2)
        corr_matrix[i, j] = NaN
    end
end

corr_matrix = corr_matrix[2:end, 1:end-1]

# Plot correlogram
AxFCorr = Axis(
    F_Subfig[1,1],
    xticks = (1:4, ["FR score", "zBMI", "age", "sex"]),
    yticks = (1:4, ["zBMI", "age", "sex", "Maternal\nEducation"]),
    aspect = 1,
    yreversed = true
)

corr_bins = collect(-1.0:0.1:1.0)
corr_cmap = :coolwarm
corr_cgrad = cgrad(corr_cmap, corr_bins, categorical=false, rev=false)

hm_corr = heatmap!(
    AxFCorr,
    corr_matrix',
    colormap = corr_cgrad,  # diverging color map
    colorrange = (-1, 1)
)

cbar_fr = Colorbar( F_Subfig[1,2], hm_corr, vertical = true)

for ci in CartesianIndices(corr_matrix)
    !isnan(corr_matrix[ci]) && text!(AxFCorr, string(round(corr_matrix[ci], digits=3)); position=(ci[2],ci[1]), align=(:center, :center))
end

#####
# Panels GHI - PERMANOVAs
#####

# Convert DataFrame's relevant columns to matrix
bin1_data_matrix_tivnorm = Matrix(bin1_inpts_tivnorm[:,10:end])
bin2_data_matrix_tivnorm = Matrix(bin2_inpts_tivnorm[:,10:end])
bin3_data_matrix_tivnorm = Matrix(bin3_inpts_tivnorm[:,10:end])
bin1_data_matrix_rclr = Matrix(bin1_inpts_rclr[:,10:end])
bin2_data_matrix_rclr = Matrix(bin2_inpts_rclr[:,10:end])
bin3_data_matrix_rclr = Matrix(bin3_inpts_rclr[:,10:end])

# Calculate the Distance Matrices
manhattan_dist = pairwise(Distances.Cityblock(), dist_data_standardized; dims = 1)
correlation_dist = pairwise(Distances.CorrDist(), dist_data_standardized; dims = 1)

rclr_data_matrix = Matrix(final_inputs[:,9:end])
euclidean_dist = pairwise(Distances.Euclidean(), rclr_data_matrix; dims = 1)

# Apply MDS for ordination
manhattan_MDS_results = fit(MDS, manhattan_dist; maxoutdim = 10, distances=true)
correlation_MDS_results = fit(MDS, correlation_dist; maxoutdim = 10, distances=true)
euclidean_MDS_results = fit(MDS, euclidean_dist; maxoutdim = 10, distances=true)

commlabels = ["Brain\n(Manhattan)", "Brain\n(Correlation)", "Brain\n(rCLR-Euclidean)"]
mdlabels = [ "FR Score", "zBMI", "Age", "Sex", "Maternal\nEducation"]

pmn_all = permanovas(
    [ manhattan_dist, correlation_dist, euclidean_dist ],
    [
        final_inputs.fr_score,
        final_inputs.zBMI,
        final_inputs.fr_score_age_years,
        final_inputs.sex,
        final_inputs.mom_edu_cat,
    ];
    n = 10000,
    commlabels = commlabels,
    mdlabels = mdlabels
)

# pmn_mantahhan = permanovas(
#     [ manhattan_dist, correlation_dist ],
#     [
#         final_inputs.fr_score,
#         final_inputs.zBMI,
#         final_inputs.fr_score_age_years,
#         final_inputs.sex,
#         final_inputs.mom_edu_cat,
#     ];
#     n = 1000,
#     commlabels = commlabels,
#     mdlabels = mdlabels
# )

# pmn_correlation  = permanovas(
#     [ correlation_dist ],
#     [
#         final_inputs.fr_score,
#         final_inputs.zBMI,
#         final_inputs.fr_score_age_years,
#         final_inputs.sex,
#         final_inputs.mom_edu_cat,
#     ]; commlabels, mdlabels
# )

PmnAx = Axis(
    D_Subfig[1, 1];
    xlabel = "Distance Type",
    ylabel = "Metadatum",
    xticklabelsize = 16,
    yticklabelsize = 16,
    title = "PERMANOVAs",
)

plot_permanovas!(PmnAx, pmn_all)

######
# Layout
######

rowsize!(fig.layout, 1, Relative(0.2))
rowsize!(fig.layout, 2, Relative(0.4))
rowsize!(fig.layout, 3, Relative(0.4))
rowgap!(A_Subfig, Relative(0.025))
colsize!(BCD_Subfig, 1, Relative(0.4))
colsize!(BCD_Subfig, 2, Relative(0.3))
colsize!(BCD_Subfig, 3, Relative(0.3))

#####
# Panel E/F/G - MDS colorby
#####

AxC_sc1 = Axis(
    C_Subfig[1,1],
    title="Manhattan Distances",
    aspect = 1,
    xlabel="MDS Dimension 1", ylabel="MDS Dimension 2")
sc1 = scatter!(
    AxC_sc1, manhattan_MDS_results.U[:, 1], manhattan_MDS_results.U[:, 2],
    color = final_inputs.fr_score_age_years,
    colormap = age_cgrad)

AxD_sc2 = Axis(
    D_Subfig[1,1],
    title="Manhattan Distances",
    aspect = 1,
    xlabel="MDS Dimension 1", ylabel="MDS Dimension 2")
sc2 = scatter!(
    AxD_sc2, manhattan_MDS_results.U[:, 1], manhattan_MDS_results.U[:, 2],
    color = final_inputs.fr_score,
    colormap = fr_cgrad)






## Saving Figure 01
fig
save(joinpath("manuscript", "figures", "Figure01.png"), fig)
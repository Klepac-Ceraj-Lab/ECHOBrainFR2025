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
input_mdata, input_brain_raw = load_brain_data()
include("notebooks/90-preprocess_brain_data.jl")

#####
# Creating Figure 1
#####

fig = Figure( size = (1600, 1000))

AB_Subfig = GridLayout(fig[1,1]; alignmomde = Outside())
A_Subfig = GridLayout(AB_Subfig[1,1]; alignmomde = Outside())
B_Subfig = GridLayout(AB_Subfig[1,2]; alignmomde = Outside())

CDE_Subfig = GridLayout(fig[2,1]; alignmomde = Outside())
C_Subfig = GridLayout(CDE_Subfig[1,1]; alignmomde = Outside())
D_Subfig = GridLayout(CDE_Subfig[1,2]; alignmomde = Outside())
E_Subfig = GridLayout(CDE_Subfig[1,3]; alignmomde = Outside())

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
# Panel A - Heatmaps
#####

hm_plotdf = sort(final_inputs, :fr_score)

AxHm_fr = Axis(
    A_Subfig[1,1],
    xlabel = "Samples, ordered by FR score",
    ylabel = "FR Score",
    ylabelrotation = 0.0)

AxHm_bmi = Axis(
    A_Subfig[2,1],
    xlabel = "Samples, ordered by FR score",
    ylabel = "zBMI (WHO)",
    ylabelrotation = 0.0)

AxHm_age = Axis(
    A_Subfig[3,1],
    xlabel = "Samples, ordered by FR score",
    ylabel = "Age (Years)",
    ylabelrotation = 0.0)

AxHm_sex = Axis(
    A_Subfig[4,1],
    xlabel = "Samples, ordered by FR score",
    ylabel = "Sex",
    ylabelrotation = 0.0)

AxHm_edu = Axis(
    A_Subfig[5,1],
    xlabel = "Samples, ordered by FR score",
    ylabel = "College Education\n(Yes/No)",
    ylabelrotation = 0.0)

linkxaxes!(AxHm_fr, AxHm_bmi)
linkxaxes!(AxHm_fr, AxHm_age)
linkxaxes!(AxHm_fr, AxHm_sex)
linkxaxes!(AxHm_fr, AxHm_edu)

hidexdecorations!(AxHm_fr)
hideydecorations!(AxHm_fr, label = false)
hidexdecorations!(AxHm_bmi)
hideydecorations!(AxHm_bmi, label = false)
hidexdecorations!(AxHm_age)
hideydecorations!(AxHm_age, label = false)
hidexdecorations!(AxHm_age)
hideydecorations!(AxHm_sex, label = false)
hidexdecorations!(AxHm_sex)
hideydecorations!(AxHm_sex, label = false)
hidexdecorations!(AxHm_edu, label = false)
hideydecorations!(AxHm_edu, label = false)

hm_fr = heatmap!(
    AxHm_fr,
    reshape(hm_plotdf.fr_score, :, 1),
    colormap = fr_cgrad
)

hm_bmi = heatmap!(
    AxHm_bmi,
    reshape(hm_plotdf.zBMI, :, 1),
    colormap = bmi_cgrad
)

hm_age = heatmap!(
    AxHm_age,
    reshape(hm_plotdf.fr_score_age_years, :, 1),
    colormap = age_cgrad
)

hm_sex = heatmap!(
    AxHm_sex,
    reshape(hm_plotdf.sex, :, 1),
    colormap = sex_cgrad
)

hm_edu = heatmap!(
    AxHm_edu,
    reshape(hm_plotdf.mom_edu_cat, :, 1),
    colormap = edu_cgrad
)

cbar_fr = Colorbar( A_Subfig[1,2], hm_fr, vertical = true)
cbar_bmi = Colorbar( A_Subfig[2,2], hm_bmi, vertical = true)
cbar_age = Colorbar( A_Subfig[3,2], hm_age, vertical = true)
cbar_sex = Colorbar( A_Subfig[4,2], hm_sex, vertical = true)
cbar_edu = Colorbar( A_Subfig[5,2], hm_edu, vertical = true)

cbar_bmi.ticks = ([-2.5, 0.0, 2.5, 5.0], ["-2.5", "0.0", "2.5", "5.0"])
cbar_age.ticks = ([1,3,5,7,9,11,13,15], ["1","3","5","7","9","11","13","15"])
cbar_sex.ticks = ([1.25, 1.75], ["Male", "Female"])
cbar_edu.ticks = ([1.25, 1.75], ["Yes", "No"])

#####
# Panel B - Scatterplots
#####

AxMSCTopHist = Axis(
    B_Subfig[1,1],
    xticks = 0.5:0.5:5.0,
    xlabel = "Food Responsiveness score",
    ylabel = "Absolute frequency")
AxMSCScatter = Axis(
    B_Subfig[2,1],
    xticks = 0.5:0.5:5.0,
    yticks = -2.5:0.5:4.5,
    xlabel = "Food Responsiveness score",
    ylabel = "WHO zBMI")
AxMSCRightHist = Axis(
    B_Subfig[2,2],
    yticks = -2.5:0.5:4.5,
    xlabel = "Absolute frequency",
    ylabel = "WHO zBMI"
)

xlims!(AxMSCTopHist, [0.49, 5.01])
tightlimits!(AxMSCTopHist, Bottom())
xlims!(AxMSCScatter, [0.49, 5.01])
ylims!(AxMSCScatter, [-2.51, +4.51])
ylims!(AxMSCRightHist, [-2.51, +4.51])
tightlimits!(AxMSCRightHist, Left())

linkxaxes!(AxMSCTopHist, AxMSCScatter)
linkyaxes!(AxMSCRightHist, AxMSCScatter)
hidexdecorations!(AxMSCTopHist)
hideydecorations!(AxMSCTopHist, ticks = false)
hidexdecorations!(AxMSCRightHist, ticks = false)
hideydecorations!(AxMSCRightHist)
hidedecorations!(AxMSCScatter, label = false, ticks = false, ticklabels = false)

sc = scatter!(
    AxMSCScatter,
    final_inputs.fr_score,
    final_inputs.zBMI,
    color = final_inputs.sex,
    colormap = :seismic,
    marker = [ ( (el == 1) ? (:rect) : (:utriangle) ) for el in final_inputs.mom_edu_cat ], # 1 is circle, 2 is rectangle
    markersize = [ log2(2*el)^2 for el in final_inputs.fr_score_age_years ]
)

hstp = hist!(
    AxMSCTopHist,
    final_inputs.fr_score,
    bins = fr_bins,
    color = collect(fr_cgrad_discrete),
    strokecolor = :black,
    strokewidth = 1
)

hsri = hist!(
    AxMSCRightHist,
    final_inputs.zBMI,
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
# Panel C/D - MDS decomposition
#####

include("notebooks/02S-FigureS1.jl")

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

# #####
# # Panel E - PERMANOVAs
# #####

commlabels = ["Brain\n(Manhattan)", "Brain\n(Correlation)"]
mdlabels = [ "FR Score", "zBMI", "Age", "Sex", "Maternal\nEducation"]

pmn_all = permanovas(
    [ manhattan_dist, correlation_dist ],
    [
        final_inputs.fr_score,
        final_inputs.zBMI,
        final_inputs.fr_score_age_years,
        final_inputs.sex,
        final_inputs.mom_edu_cat,
    ];
    n = 1000,
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
    E_Subfig[1, 1];
    xlabel = "Distance Type",
    ylabel = "Metadatum",
    xticklabelsize = 16,
    yticklabelsize = 16,
    title = "PERMANOVAs",
)

plot_permanovas!(PmnAx, pmn_all)

## Saving Figure 01
fig
save(joinpath("manuscript", "figures", "Figure01.png"), fig)
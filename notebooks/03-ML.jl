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
using JLD2
using DecisionTree              # Needed to extract feature importances from the classical forest
using XGBoost                   # Needed to extract feature importances from the booster
using MLJDecisionTreeInterface  # Provides the RandomForesRegressor for MLJ
using MLJXGBoostInterface       # Provides the XGBoostRegressor for MLJ
using ECHOBrainFR2025

## Loading Data
input_mdata, input_brain_raw = load_brain_data()
include("notebooks/90-preprocess_brain_data.jl")

# include("src/ProbeElasticNet.jl")
# include("src/ProbeRFRegressor.jl")
include("src/learning.jl")

for (this_col_idxes, this_model_name) in zip(
    [
        collect(5:ncol(final_inputs)),
        collect(6:ncol(final_inputs)),
        collect(7:ncol(final_inputs)),
        collect(9:ncol(final_inputs))
        # collect(5:ncol(final_inputs_rclr)),
        # collect(6:ncol(final_inputs_rclr)),
        # collect(7:ncol(final_inputs_rclr)),
        # collect(9:ncol(final_inputs_rclr))
    ],
    [
        "regression_FR_zBMI_SES_AgeSex_Brain",
        "regression_FR_SES_AgeSex_Brain",
        "regression_FR_AgeSex_Brain",
        "regression_FR_OnlyBrain"
    ]
    )

    # println(this_col_idxes,this_model_name)

    # ## 1. ElasticNet

    enet_obj = probe_regression_elasticnet(
        "enet_test",
        final_inputs,
        # final_inputs_rclr,
        identity,
        this_col_idxes,
        :fr_score;
        unique_col = :UID,
        n_folds = 5,
        n_replicas = 20,
        n_rngs = 5,
        verbose = true,
        train_rng=Random.GLOBAL_RNG
    )

    report_regression_merits(enet_obj)
    enet_imps = hpimportances(enet_obj, 1)
    filtered_enet_imps = subset(enet_imps, :weightedImportance => ( x -> abs.(x) .> 0.0001 ))
    JLD2.@save joinpath(joinpath("results", this_model_name*"_enet_model.jld")) enet_obj

    ## 2. Classical RandomForest with filtered inputs

    rf_obj = probe_regression_randomforest(
        "rf_test",
        final_inputs,
        # final_inputs_rclr,
        identity,
        this_col_idxes,
        :fr_score;
        custom_input_group = filtered_enet_imps.variable,
        unique_col = :UID,
        n_folds = 5,
        n_replicas = 10,
        n_rngs = 3,
        verbose = true,
        tuning_space = (
            maxnodes_range = [ -1, 5 ],
            nodesize_range = [ 7, 11 ],
            min_samples_split = [ 2, 5 ],
            sampsize_range =  [ 0.8 ],
            mtry_range = [ -1, 0, 5, 10, 12, 15],
            # mtry_range = [ -1, 0, 5, 10, 12, 15, 20],
            ntrees_range = [ 128 ]
        ),
        train_rng=Random.GLOBAL_RNG
    )

    @show report_regression_merits(rf_obj)
    CSV.write(joinpath("results", this_model_name*"_rf_merits.csv"), report_regression_merits(rf_obj))
    JLD2.@save joinpath(joinpath("results", this_model_name*"_rf_model.jld")) rf_obj

    hp_idx = sort(report_regression_merits(rf_obj), :Val_RMSE_mean).Hyperpar_Idx[1]

    rf_imps = hpimportances(rf_obj, hp_idx)
    rf_imps.cumsum = cumsum(rf_imps.weightedImportance)
    # nfeat_toplot = findfirst(rf_imps.cumsum .> 0.7)
    nfeat_toplot = nrow(rf_imps)
    rf_imps.correl = [ cor(final_inputs[:, ccol], final_inputs.fr_score) for ccol in rf_imps.variable ]
    # rf_imps.correl = [ cor(final_inputs_rclr[:, ccol], final_inputs_rclr.fr_score) for ccol in rf_imps.variable ]
    rf_imps.impsign = rf_imps.weightedImportance .* sign.(rf_imps.correl)
    CSV.write(joinpath("results", this_model_name*"_rf_importances.csv"), rf_imps)

    fig = Figure(size = (600,1000))

    ax = Axis(
        fig[1,1],
        yticks = (1:nfeat_toplot, rf_imps.variable[1:nfeat_toplot]),
        yreversed = true
    )

    bp = barplot!(
        ax,
        1:nfeat_toplot,
        rf_imps.weightedImportance[1:nfeat_toplot],
        direction = :x
    )

    save(joinpath("results", this_model_name*"_rf_importances_abs.png"), fig)

    impsign_rf_impos = rf_imps[1:nfeat_toplot, :]
    sort!(impsign_rf_impos, :impsign; rev = true)

    fig = Figure(size = (600,1000))

    ax = Axis(
        fig[1,1],
        yticks = (1:nfeat_toplot, impsign_rf_impos.variable[1:nfeat_toplot]),
        yreversed = true
    )

    bp = barplot!(
        ax,
        1:nfeat_toplot,
        impsign_rf_impos.impsign[1:nfeat_toplot],
        color = [ ( ( el < 0.0) ? :deeppink2 : :blue ) for el in impsign_rf_impos.impsign[1:nfeat_toplot] ],
        direction = :x
    )

    save(joinpath("results", this_model_name*"_rf_importances_ladder.png"), fig)

    # xgb_obj = probe_regression_xgboost(
    #     "xgboost_test",
    #     final_inputs,
    #     identity,
    #     collect(5:ncol(final_inputs)),
    #     :fr_score;
    #     # custom_input_group = filtered_enet_imps.variable,
    #     unique_col = :UID,
    #     n_folds = 5,
    #     n_replicas = 10,
    #     n_rngs = 3,
    #     verbose = true,
    #     tuning_space = (;
    #         max_depth = [2, 3],
    #         eta = [0.001, 0.01],
    #         num_round = [50, 100],
    #         gamma = [0.5, 1.0],
    #         subsample = [0.5, 0.7],
    #         colsample_bytree = [0.5, 0.7],
    #         reg_lambda = [1.0, 5.0, 10.0],
    #         reg_alpha = [0.1, 0.5, 0.9]
    #     ),
    #     # tuning_space = (;
    #     # max_depth         = [2, 3],          # shallower trees
    #     # eta               = [0.001, 0.005, 0.01], # lower learning rate
    #     # num_round         = [50, 75, 100],        # number of boosting rounds
    #     # gamma             = [0.5, 1.0, 2.0],      # stronger split criteria
    #     # subsample         = [0.7],           # more randomness in row sampling
    #     # colsample_bytree  = [0.7],           # more randomness in feature sampling
    #     # reg_lambda        = [1.0, 5.0, 10.0, 20.0],    # L2 regularization
    #     # reg_alpha         = [0.1, 0.5, 0.9]       # L1 regularization
    #     # ),
    #     train_rng=Random.GLOBAL_RNG
    # )

    # @show reg_merits = subset(report_regression_merits(xgb_obj), :Val_Cor_mean => x -> .!(isnan.(x)))
    # CSV.write(joinpath("results", this_model_name*"_xgb_merits.csv"), report_regression_merits(rf_obj))
    # xgb_imps = hpimportances(xgb_obj, 1)
    # sort!(xgb_imps, :weightedImportance; rev=true)
    # CSV.write(joinpath("results", this_model_name*"_xgb_imps.csv"), xgb_imps)

end
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
using ECHOBrainFR2025

## Loading Data
input_mdata, input_brain_raw, _, _ = load_brain_data()
include("notebooks/90-preprocess_brain_data.jl")

# include("src/ProbeElasticNet.jl")
# include("src/ProbeRFRegressor.jl")
# include("src/learning.jl")

for (this_col_idxes, this_model_name) in zip(
    [
        collect(6:ncol(final_inputs)),
        collect(7:ncol(final_inputs)),
        collect(8:ncol(final_inputs)),
        collect(10:ncol(final_inputs))
        # collect(6:ncol(final_inputs_rclr)),
        # collect(7:ncol(final_inputs_rclr)),
        # collect(8:ncol(final_inputs_rclr)),
        # collect(10:ncol(final_inputs_rclr))
    ],
    [
        "regression_FR_zBMI_SES_AgeSex_Brain",
        "regression_FR_SES_AgeSex_Brain",
        "regression_FR_AgeSex_Brain",
        "regression_FR_OnlyBrain"
    ]
    )

    perform_ML_cals("ALLPTS", this_col_idxes, this_model_name, final_inputs)
    # perform_ML_cals("ALLPTS", this_col_idxes, this_model_name, final_inputs_rclr)
end

for this_bin in [1,2,3]

    bin_inpts = subset(final_inputs, :agebin => x -> x .== this_bin)
    # bin_inpts = subset(final_inputs_rclr, :agebin => x -> x .== this_bin)
    println("bin_inpts has $(nrow(bin_inpts)) rows!")

    for (this_col_idxes, this_model_name) in zip(
        [
            collect(6:ncol(bin_inpts)),
            collect(7:ncol(bin_inpts)),
            collect(8:ncol(bin_inpts)),
            collect(10:ncol(bin_inpts))
            # collect(6:ncol(final_inputs_rclr)),
            # collect(7:ncol(final_inputs_rclr)),
            # collect(8:ncol(final_inputs_rclr)),
            # collect(10:ncol(final_inputs_rclr))
        ],
        [
            "regression_FR_zBMI_SES_AgeSex_Brain",
            "regression_FR_SES_AgeSex_Brain",
            "regression_FR_AgeSex_Brain",
            "regression_FR_OnlyBrain"
        ]
        )

        perform_ML_cals("BIN$(string(this_bin))", this_col_idxes, this_model_name, bin_inpts)
    end
end
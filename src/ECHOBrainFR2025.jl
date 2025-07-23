module ECHOBrainFR2025

    include("load_data.jl")
    include("learning.jl")
    include("utils.jl")

    export load_brain_data

    export CVResults
    export partitionvec
    export run_cv_workflow
    export build_elasticnet_model, predict_elasticnet
    export build_rf_model, predict_rf
    export build_xgboost_model, predict_xgboost
    export probe_regression_elasticnet
    export probe_regression_randomforest
    export probe_regression_xgboost
    export conf_interval
    export report_regression_merits
    export hpimportances
    export force_continuous

    export robust_geometric_mean_nonzero, rclr_vector
    export autonormalize, rangenormalize
    export ordered_brain_names, ordered_brain_renamepairs, ordered_brain_newnames
end

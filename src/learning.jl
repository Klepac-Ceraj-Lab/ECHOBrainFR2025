using DataFrames, Random, Statistics, GLMNet, MLJ, MLJDecisionTreeInterface, GLM
using Chain  # for the @chain macro

# A simple struct for returning results
struct CVResults
    report_df::DataFrame
    importances::DataFrame
    saved_predictions::DataFrame
    saved_pertinences::DataFrame
    external_predictions::Union{Nothing, DataFrame}
end

# --- Common helper functions ---

# Partition vector into folds (unchanged from your version)
function partitionvec(x, stride, i, longtail::Bool=true)
    stride > 0 || error("stride must be positive")
    starts = firstindex(x):stride:(lastindex(x)-longtail*stride)+1
    subvecs = [view(x, starts[i]:get(starts, i+1, lastindex(x)+1)-1) for i in eachindex(starts)]
    train = reduce(vcat, [collect(subvecs[j]) for j in eachindex(subvecs) if j != i])
    test  = collect(subvecs[i])
    return train, test
end

# Example metric function (you can add others as needed)
# rmse(yhat, y) = sqrt(mean((yhat .- y).^2)) ## Already in MLJ

# --- Common CV workflow ---
function run_cv_workflow(
    original_df::DataFrame,
    data_preparation_fun::Function,
    input_cols,
    output_col;
    model_type::Symbol,  # e.g. :elasticnet or :randomforest
    build_model_fun::Function,  # should return (model, train_y_hat, val_y_hat, importance)
    predict_fun::Function,      # function(model, X_df) -> predictions (for external data)
    tuning_space = nothing,
    n_folds = 10,
    n_replicas = 10,
    n_rngs = 100,
    ext_df::Union{DataFrame, Nothing} = nothing,
    unique_col = :sample,
    custom_input_group = nothing,
    split_strat = nothing,
    train_rng = Random.GLOBAL_RNG,
    verbose = true
)
    # 1. Data preparation and external data harmonization
    prediction_df = data_preparation_fun(original_df)
    input_cols = collect(input_cols[1]:ncol(prediction_df))
    if !isnothing(ext_df)
        @assert length(ext_df[:, unique_col]) == length(unique(ext_df[:, unique_col]))
        for col in names(prediction_df[:, input_cols])
            if !(col ∈ names(ext_df))
                insertcols!(ext_df, col => 0.0)
            end
        end
        select!(ext_df, names(prediction_df))
    end
    verbose && println("Data prepared.")

    # 2. Build tuning grid if not provided
    if isnothing(tuning_space)
        if model_type == :elasticnet
            # tuning_space = (aa = 10 .^ collect(-6:0.05:0), ll = collect(0.01:0.05:19.99))
            # tuning_space = (aa = 10 .^ collect(-6:0.5:0), ll = collect(0.01:0.5:19.99))
            tuning_space = (aa = [ 0.5 ], ll = [ 10.0 ])
        elseif model_type == :randomforest
            tuning_space = (maxnodes_range = [-1],
                            nodesize_range = [5],
                            min_samples_split = [2],
                            sampsize_range = [0.7],
                            mtry_range = [-1],
                            ntrees_range = [10])
        end
    end
    tuning_grid = vec(collect(Base.product(values(tuning_space)...)))
    
    # 3. Precompute fold partitions (simple partitioning by index)
    dataset_partitions = [ partitionvec(collect(1:nrow(prediction_df)),
                                        floor(Int, nrow(prediction_df) / n_folds),
                                        fold_idx, true)
                           for fold_idx in 1:n_folds ]
    verbose && println("Partitions created.")

    # 4. Preallocate arrays (total number of models to be fitted)
    n_models = n_folds * n_replicas * n_rngs * length(tuning_grid)
    model_colnames = Symbol.("model_" .* lpad.(string.(1:n_models), 6, "0"))
    hyperpar_idxes = Vector{Int}(undef, n_models)
    replica_idxes = similar(hyperpar_idxes)
    fold_idxes = similar(hyperpar_idxes)
    rng_idxes = similar(hyperpar_idxes)
    train_rmses = Vector{Float64}(undef, n_models)
    val_rmses = similar(train_rmses)
    test_rmses = similar(train_rmses)
    train_mapes = similar(train_rmses)
    val_mapes = similar(train_rmses)
    test_mapes = similar(train_rmses)
    train_cors = similar(train_rmses)
    val_cors = similar(train_rmses)
    test_cors = similar(train_rmses)
    saved_input_predictions = Matrix{Float64}(undef, nrow(prediction_df), n_models)
    saved_input_pertinences = Matrix{String}(undef, nrow(prediction_df), n_models)
    saved_ext_predictions = isnothing(ext_df) ? nothing :
        Matrix{Float64}(undef, nrow(ext_df), n_models)
    n_features = isnothing(custom_input_group) ? length(input_cols) : length(custom_input_group)
    importances = Matrix{Float64}(undef, n_features, n_models)

    model_index = 0

    # 5. CV loop over hyperparameters, replicas, folds and RNG seeds
    for (hp_idx, hp) in enumerate(tuning_grid)
        for rep_idx in 1:n_replicas
            Random.seed!(train_rng, rep_idx-1)
            shuffled_df = prediction_df[Random.randperm(train_rng, nrow(prediction_df)), :]
            X = isnothing(custom_input_group) ? shuffled_df[:, input_cols] : shuffled_df[:, custom_input_group]
            y = shuffled_df[:, output_col]
            verbose && println("Dataset shuffled for replica $(rep_idx) of hp index $(hp_idx).")
            for (fold_number, (train_idx, val_idx)) in enumerate(dataset_partitions)
                for rng_idx in 1:n_rngs
                    Random.seed!(train_rng, rng_idx-1)
                    # Build and fit the model on the training data
                    model, train_y_hat, val_y_hat, imp = build_model_fun(X, y, train_idx, val_idx, hp, train_rng, verbose)
                    
                    # Compute metrics on training and validation sets
                    tr_rmse  = rmse(train_y_hat, y[train_idx])
                    tr_mape  = mean(abs.(train_y_hat .- y[train_idx]) ./ y[train_idx])
                    tr_cor   = Statistics.cor(train_y_hat, y[train_idx])
                    v_rmse   = rmse(val_y_hat, y[val_idx])
                    v_mape   = mean(abs.(val_y_hat .- y[val_idx]) ./ y[val_idx])
                    v_cor    = Statistics.cor(val_y_hat, y[val_idx])
                    
                    # External test set predictions (if provided)
                    if !isnothing(ext_df)
                        test_y_hat = predict_fun(model, ext_df)
                        t_rmse = rmse(test_y_hat, ext_df[:, output_col])
                        t_mape = mean(abs.(test_y_hat .- ext_df[:, output_col]) ./ ext_df[:, output_col])
                        t_cor  = Statistics.cor(test_y_hat, ext_df[:, output_col])
                        saved_ext_predictions[:, model_index+1] = test_y_hat
                    else
                        t_rmse = NaN; t_mape = NaN; t_cor = NaN
                    end

                    model_index += 1
                    colname = model_colnames[model_index]
                    importances[:, model_index] = imp

                    # Save predictions and pertinence info for train + validation samples
                    input_predictions = @chain DataFrame(
                        unique_col => vcat(shuffled_df[train_idx, unique_col],
                                            shuffled_df[val_idx, unique_col]),
                        colname => map(x -> round(x; digits=1), vcat(train_y_hat, val_y_hat))
                    ) begin
                        sort(unique_col)
                    end
                    saved_input_predictions[:, model_index] = input_predictions[!, colname]

                    input_pertinences = @chain DataFrame(
                        unique_col => vcat(shuffled_df[train_idx, unique_col],
                                            shuffled_df[val_idx, unique_col]),
                        colname => vcat(fill("train", length(train_idx)),
                                         fill("val", length(val_idx)))
                    ) begin
                        sort(unique_col)
                    end
                    saved_input_pertinences[:, model_index] = input_pertinences[!, colname]

                    # Save model metadata and metrics
                    hyperpar_idxes[model_index] = hp_idx
                    replica_idxes[model_index] = rep_idx
                    fold_idxes[model_index] = fold_number
                    rng_idxes[model_index] = rng_idx
                    train_rmses[model_index] = tr_rmse
                    val_rmses[model_index]   = v_rmse
                    test_rmses[model_index]  = t_rmse
                    train_mapes[model_index] = tr_mape
                    val_mapes[model_index]   = v_mape
                    test_mapes[model_index]  = isnothing(ext_df) ? NaN : t_mape
                    train_cors[model_index]  = tr_cor
                    val_cors[model_index]    = v_cor
                    test_cors[model_index]   = t_cor
                    
                    verbose && println("Model $(model_index) evaluated.")
                end
            end
        end
    end

    # 6. Compile report and final DataFrames
    report_df = DataFrame(
        Hyperpar_Idx = hyperpar_idxes,
        Hyperpars = tuning_grid[hyperpar_idxes],
        Replica_Idx  = replica_idxes,
        Fold_Idx     = fold_idxes,
        Rng_Idx      = rng_idxes,
        Train_RMSE   = train_rmses,
        Val_RMSE     = val_rmses,
        Test_RMSE    = test_rmses,
        Train_MAPE   = train_mapes,
        Val_MAPE     = val_mapes,
        Test_MAPE    = test_mapes,
        Train_Cor    = train_cors,
        Val_Cor      = val_cors,
        Test_Cor     = test_cors
    )

    variables = isnothing(custom_input_group) ? names(prediction_df[:, input_cols]) : custom_input_group
    final_importances = hcat(DataFrame(variable = variables),
                             DataFrame(importances, :auto) |> (df -> rename!(df, names(df) .=> model_colnames)))
    final_saved_predictions = hcat(DataFrame(unique_col => sort(prediction_df[:, unique_col])),
                                   DataFrame(saved_input_predictions, :auto) |> (df -> rename!(df, names(df) .=> model_colnames)))
    final_saved_pertinences = hcat(DataFrame(unique_col => sort(prediction_df[:, unique_col])),
                                   DataFrame(saved_input_pertinences, :auto) |> (df -> rename!(df, names(df) .=> model_colnames)))
    final_external_predictions = isnothing(ext_df) ? nothing :
        hcat(DataFrame(unique_col => sort(ext_df[:, unique_col])),
             DataFrame(saved_ext_predictions, :auto) |> (df -> rename!(df, names(df) .=> model_colnames)))
    
    return CVResults(report_df, final_importances, final_saved_predictions, final_saved_pertinences, final_external_predictions)
end

# --- Model-specific builder functions ---

# For ElasticNet feature selection:
function build_elasticnet_model(X, y, train_idx, val_idx, hp, rng, verbose)
    # hp is assumed to be a tuple (alpha, lambda)
    # enet_model = glmnet(Matrix(X[train_idx, :]), y[train_idx], alpha = hp[1], lambda = [hp[2]])
    enet_model = glmnetcv(Matrix(X[train_idx, :]), y[train_idx], alpha = 0.5)
    verbose && println("ElasticNet model fitted.")
    train_y_hat = vec(GLMNet.predict(enet_model, Matrix(X[train_idx, :])))
    val_y_hat   = vec(GLMNet.predict(enet_model, Matrix(X[val_idx, :])))
    # Use the beta coefficients as feature importances
    # importance = enet_model.betas
    importance = GLMNet.coef(enet_model)
    return enet_model, train_y_hat, val_y_hat, importance
end

# Prediction function for ElasticNet (if needed externally)
function predict_elasticnet(model, df)
    return vec(GLMNet.predict(model, Matrix(df)))
end

# For Random Forest regression:
function build_rf_model(X, y, train_idx, val_idx, hp, rng, verbose)
    rf_model = MLJDecisionTreeInterface.RandomForestRegressor(
        max_depth         = hp[1],
        min_samples_leaf  = hp[2],
        min_samples_split = hp[3],
        sampling_fraction = hp[4],
        n_subfeatures     = hp[5],
        n_trees           = hp[6],
        rng               = rng
    )
    rf_machine = machine(rf_model, X[train_idx, :], y[train_idx], cache=false)
    MLJ.fit!(rf_machine, verbosity=0)
    verbose && println("Random Forest model fitted.")
    # Predict and apply slope correction for both train and validation:
    train_y_hat = MLJ.predict(rf_machine, X[train_idx, :])
    slope_model_train = lm(@formula(y ~ yhat + 1),
                           DataFrame(y = y[train_idx], yhat = train_y_hat))
    train_y_hat = GLM.predict(slope_model_train, DataFrame(yhat = MLJ.predict(rf_machine, X[train_idx, :])))
    
    val_y_hat = MLJ.predict(rf_machine, X[val_idx, :])
    slope_model_val = lm(@formula(y ~ yhat + 1),
                         DataFrame(y = y[val_idx], yhat = val_y_hat))
    val_y_hat = GLM.predict(slope_model_val, DataFrame(yhat = MLJ.predict(rf_machine, X[val_idx, :])))
    
    # Use the impurity-based importance
    importance = impurity_importance(rf_machine.fitresult)
    return rf_machine, train_y_hat, val_y_hat, importance
end

# Prediction function for Random Forest (for external test set)
function predict_rf(model, ext_df)
    return MLJ.predict(model, ext_df)
end

# Function to build and fit an XGBoost model
function build_xgboost_model(X, y, train_idx, val_idx, hp, rng, verbose)
    # hp is assumed to be a tuple with the following order:
    # (max_depth, eta, num_round, gamma, subsample, colsample_bytree)
    # For example: hp = (max_depth, eta, num_round, gamma, subsample, colsample_bytree)
    model = XGBoostRegressor(
        max_depth         = hp[1],
        eta               = hp[2],
        num_round         = hp[3],
        gamma             = hp[4],
        subsample         = hp[5],
        colsample_bytree  = hp[6],
        lambda        = hp[7],
        alpha         = hp[8]
    )
    
    # Create and fit the MLJ machine on training data only
    xgb_machine = machine(model, force_continuous(X[train_idx, :]), y[train_idx])
    MLJ.fit!(xgb_machine, verbosity=0)
    verbose && println("XGBoost model fitted.")
    
    # Obtain predictions on train and validation sets
    train_y_hat = MLJ.predict(xgb_machine, force_continuous(X[train_idx, :]))
    val_y_hat   = MLJ.predict(xgb_machine, force_continuous(X[val_idx, :]))
    
    # Extract feature importance from the fitted booster.
    # The underlying XGBoost model can be extracted as follows:
    booster = xgb_machine.fitresult  # This is the trained XGBoost Booster
    # XGBoost.importance returns a Dict with keys like "f0", "f1", etc.
    imp_dict = XGBoost.importance(booster[1])
    n_features = size(X, 2)
    importance = zeros(n_features)
    for i in 1:n_features
        key = "f$(i-1)"  # XGBoost features are named f0, f1, ...
        importance[i] = get(imp_dict, key, 0.0)
    end
    
    return xgb_machine, train_y_hat, val_y_hat, importance
end

# Prediction function for XGBoost
function predict_xgboost(model, df)
    return MLJ.predict(model, df)
end

# --- High-level functions for each workflow ---
function probe_regression_elasticnet(name::String, original_df::DataFrame, data_preparation_fun::Function, input_cols, output_col; kwargs...)
    cv_results = run_cv_workflow(original_df, data_preparation_fun, input_cols, output_col;
                                 model_type = :elasticnet,
                                 build_model_fun = build_elasticnet_model,
                                 predict_fun = predict_elasticnet,
                                 kwargs...)
    # Here you can wrap cv_results into your ProbeData struct if desired
    return cv_results
end

function probe_regression_randomforest(name::String, original_df::DataFrame, data_preparation_fun::Function, input_cols, output_col; kwargs...)
    cv_results = run_cv_workflow(original_df, data_preparation_fun, input_cols, output_col;
                                 model_type = :randomforest,
                                 build_model_fun = build_rf_model,
                                 predict_fun = predict_rf,
                                 kwargs...)
    return cv_results
end

function probe_regression_xgboost(name::String, original_df::DataFrame, data_preparation_fun::Function, input_cols, output_col; kwargs...)
    cv_results = run_cv_workflow(
        original_df, 
        data_preparation_fun, 
        input_cols, 
        output_col;
        model_type = :xgboost,
        build_model_fun = build_xgboost_model,
        predict_fun = predict_xgboost,
        kwargs...
    )
    return cv_results
end

conf_interval(vv, critical_z=1.96) = critical_z * std(vv) / sqrt(length(vv))

report_regression_merits(m, ccol = :Val_Cor_mean) = sort(combine(
    groupby(m.report_df, :Hyperpar_Idx),
    :Hyperpars => (x -> first(x)) => :Hyperpars,
    :Train_RMSE => (x -> round(mean(x); digits = 4)) => :Train_RMSE_mean,
    :Train_RMSE => (x -> round(conf_interval(x); digits = 4)) => :Train_RMSE_CI,
    :Train_Cor => (x -> round(mean(x); digits = 4)) => :Train_Cor_mean,
    :Train_Cor => (x -> round(conf_interval(x); digits = 4)) => :Train_Cor_CI,
    :Val_RMSE => (x -> round(mean(x); digits = 4)) => :Val_RMSE_mean,
    :Val_RMSE => (x -> round(conf_interval(x); digits = 4)) => :Val_RMSE_CI,
    :Val_Cor => (x -> round(mean(x); digits = 4)) => :Val_Cor_mean,
    :Val_Cor => (x -> round(conf_interval(x); digits = 4)) => :Val_Cor_CI,
    :Test_RMSE => (x -> round(mean(x); digits = 4)) => :Test_RMSE_mean,
    :Test_RMSE => (x -> round(conf_interval(x); digits = 4)) => :Test_RMSE_CI,
    :Test_Cor => (x -> round(mean(x); digits = 4)) => :Test_Cor_mean,
    :Test_Cor => (x -> round(conf_interval(x); digits = 4)) => :Test_Cor_CI,
    # :Hyperpar_Idx => (x -> first(x)) => :Hyperpar_Idx,
), ccol; rev = true)

function hpimportances(m, hp = 1; change_hashnames=false, hashnamestable::Union{Nothing, DataFrame} = nothing)
    importances = m.importances
    subset_idx = findall(m.report_df.Hyperpar_Idx .== hp)

    mean_importances = map(x -> sum(x .* subset_idx)/sum(subset_idx), collect(eachrow(Matrix(importances[:, subset_idx .+ 1]))))
    mean_importances_df = sort( DataFrame(:variable => importances[:,1], :weightedImportance => mean_importances), :weightedImportance; rev=true)

    if change_hashnames
        newnames = leftjoin(mean_importances_df, hashnamestable, on = :variable => :hashname).longname
        mean_importances_df.variable = newnames
    end

    return mean_importances_df
end

function force_continuous(df::DataFrame)
    df_cont = DataFrame()
    for col in names(df)
        # Convert each column element to Float64, even if it was a Count type
        df_cont[!, col] = Float64.(df[!, col])
    end
    return df_cont
end
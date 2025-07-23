function probe_regression_randomforest(
    name::String,
    original_df::DataFrame,
    data_preparation_fun::Function,
    input_cols,
    output_col;
    split_strat::Union{String, Symbol, Nothing} = nothing,
    custom_input_group::Union{Vector, Nothing} = nothing,
    ext_df::Union{DataFrame, Nothing} = nothing,
    ext_firstinputcol = 0,
    ext_uniquecol = :sample,
    unique_col = :sample,
    n_folds = 10,
    n_replicas = 10,
    n_rngs = 100,
    tuning_space = (;
        maxnodes_range = [ -1 ],
        nodesize_range = [ 5 ],
        min_samples_split = [ 2 ],
        sampsize_range = [ 0.7 ],
        mtry_range = [ -1 ],
        ntrees_range = [ 10 ]
        ),
    verbose = true,
    train_rng=Random.GLOBAL_RNG
    )

    @assert length(original_df[:, unique_col]) == length(unique(original_df[:, unique_col]))

    # ## 1. Subsetting Data and shuffling DataFrame
    prediction_df = data_preparation_fun(original_df)
    input_cols = collect(input_cols[1]:ncol(prediction_df))

    # This block of code will insert input-expected columns as 0.0 in the external test dataframe,
    # given that it exists, in case they do not exist there.
    if (!isnothing(ext_df)) 
        
        @assert length(ext_df[:, ext_uniquecol]) == length(unique(ext_df[:, ext_uniquecol]))

        for col in names(prediction_df[:, input_cols]) ## column insertion code block
            if !(col ∈ names(ext_df))
                insertcols!(ext_df, col => 0.0)
            end
        end

        select!(ext_df, names(prediction_df))
    end

    verbose && print("prediction_df initialized")

    # ## 3. Building hyperparameter tuning grid
    tuning_grid = vec(collect(Base.product(tuning_space...)))
    #     tuning_space.maxnodes_range,
    #     tuning_space.nodesize_range,
    #     tuning_space.sampsize_range,
    #     tuning_space.mtry_range,
    #     tuning_space.ntrees_range
    # )))

    dataset_partitions = Vector{Tuple{Vector{Int64}, Vector{Int64}}}()
    for fold_idx in 1:n_folds
        train, val = partitionvec(collect(1:nrow(prediction_df)),floor(Int64, nrow(prediction_df)/n_folds),fold_idx, true)
        push!(dataset_partitions, (train, val))
    end

    verbose && print("partitions calculated")

    # ys = Vector{Vector{Any}}()
    model_colnames = Symbol.("model_" .* map(x -> lpad(string(x), 6, "0"), collect(1:(n_folds*n_replicas*n_rngs*length(tuning_grid)))) )
    hyperpar_idxes = Vector{Int64}(undef, n_folds*n_replicas*n_rngs*length(tuning_grid))
    replica_idxes = Vector{Int64}(undef, n_folds*n_replicas*n_rngs*length(tuning_grid))
    fold_idxes = Vector{Int64}(undef, n_folds*n_replicas*n_rngs*length(tuning_grid))
    rng_idxes = Vector{Int64}(undef, n_folds*n_replicas*n_rngs*length(tuning_grid))
    train_rmses = Vector{Float64}(undef, n_folds*n_replicas*n_rngs*length(tuning_grid))
    val_rmses = Vector{Float64}(undef, n_folds*n_replicas*n_rngs*length(tuning_grid))
    test_rmses = Vector{Float64}(undef, n_folds*n_replicas*n_rngs*length(tuning_grid))
    train_mapes = Vector{Float64}(undef, n_folds*n_replicas*n_rngs*length(tuning_grid))
    val_mapes = Vector{Float64}(undef, n_folds*n_replicas*n_rngs*length(tuning_grid))
    test_mapes = Vector{Float64}(undef, n_folds*n_replicas*n_rngs*length(tuning_grid))
    train_cors = Vector{Float64}(undef, n_folds*n_replicas*n_rngs*length(tuning_grid))
    val_cors = Vector{Float64}(undef, n_folds*n_replicas*n_rngs*length(tuning_grid))
    test_cors = Vector{Float64}(undef, n_folds*n_replicas*n_rngs*length(tuning_grid))
    saved_input_predictions = Matrix{Float64}(undef, nrow(prediction_df), n_folds*n_replicas*n_rngs*length(tuning_grid))
    saved_input_pertinences = Matrix{String}(undef, nrow(prediction_df), n_folds*n_replicas*n_rngs*length(tuning_grid))
    !isnothing(ext_df) && (saved_ext_predictions = Matrix{Float64}(undef, nrow(ext_df), n_folds*n_replicas*n_rngs*length(tuning_grid)))

    if isnothing(custom_input_group)
        importances = Matrix{Float64}(undef, length(input_cols), n_folds*n_replicas*n_rngs*length(tuning_grid))
    else
        importances = Matrix{Float64}(undef, length(custom_input_group), n_folds*n_replicas*n_rngs*length(tuning_grid))
    end

    verbose && println("Arrays initialized")

    model_index = 0

    # function getweights(x; lt=3.0, ut=12.0, lw=15, uw=15)
    # function getweights(x; lt=4.0, ut=11.0, lw=3, uw=15)
    # function getweights(x; lt=3.0, ut=15.0, lw=15, uw=15)
    function getweights(x; lt=3.0, ut=14.0, lw=15, uw=15)

        if (x <= lt)
            return lw
        elseif (x >= ut)
            return uw
        else
            return 1
        end
    end

    for hp_idx in eachindex(tuning_grid)

        for rep_idx in 1:n_replicas

            Random.seed!(train_rng, rep_idx-1)
            shuffled_df = prediction_df[Random.randperm(train_rng, nrow(prediction_df)), :]
            if isnothing(custom_input_group)
                X = shuffled_df[:, input_cols]
            else
                X = shuffled_df[:, custom_input_group]
            end
            y = shuffled_df[:, output_col]

            verbose && println("Dataset shuffled")

            for fold_idx in 1:n_folds

                if isnothing(split_strat)
                    train, val = dataset_partitions[fold_idx]
                else #This code splits subjects into train and validation sets for every
                    dataset_partitions = Vector{Tuple{Vector{Int64}, Vector{Int64}}}()
                    unique_datasources = unique(shuffled_df.datasource)
                    train_test_collect = map(
                        this_ds -> begin
                        subsetted_df = subset(shuffled_df, :datasource => x -> x .== this_ds)
                        unique_intra_subjects = unique(subsetted_df.subject_id)
                        train_subs, val_subs = partitionvec(unique_intra_subjects,floor(Int64, length(unique_intra_subjects)/n_folds),fold_idx, true)
                        train = findall(shuffled_df.subject_id .∈ Ref(train_subs) ) 
                        val = findall(shuffled_df.subject_id .∈ Ref(val_subs) )
                        return(train, val)
                        end,
                        unique_datasources
                    )
                    train = vcat([el[1] for el in train_test_collect ]...)
                    val = vcat([el[2] for el in train_test_collect ]...)
                end

                for rng_idx in 1:n_rngs
    
                    Random.seed!(train_rng, rng_idx-1)

                    rf_model = MLJDecisionTreeInterface.RandomForestRegressor(
                        max_depth = tuning_grid[hp_idx][1],
                        min_samples_leaf = tuning_grid[hp_idx][2],
                        min_samples_split = tuning_grid[hp_idx][3],
                        sampling_fraction = tuning_grid[hp_idx][4],
                        n_subfeatures = tuning_grid[hp_idx][5],
                        n_trees = tuning_grid[hp_idx][6],
                        rng = train_rng
                    )

                    verbose && println("RF_model declared")
    
                    rf_machine = machine(rf_model, X[train, :], y[train], cache=false)
                    MLJ.fit!(rf_machine, verbosity=0)

                    verbose && println("Machine built and fit")

                    train_y_hat = MLJ.predict(rf_machine, X[train, :]) 
                    slopecorrection = lm(@formula(y ~ yhat + 1), DataFrame(:y => y[train], :yhat => train_y_hat)) 
                    train_y_hat = GLM.predict(slopecorrection, DataFrame(:yhat => MLJ.predict(rf_machine, X[train, :]) ))
                    train_rmse = rmse(train_y_hat, y[train])
                    train_mape = mean(abs.(train_y_hat .- y[train]) ./ y[train])
                    train_cor = Statistics.cor(train_y_hat, y[train])

                    val_y_hat = MLJ.predict(rf_machine, X[val, :]) 
                    slopecorrection = lm(@formula(y ~ yhat + 1), DataFrame(:y => y[val], :yhat => val_y_hat)) # uncomment to learn bias correction on OOB data
                    val_y_hat = GLM.predict(slopecorrection, DataFrame(:yhat => MLJ.predict(rf_machine, X[val, :]) ))
                    val_rmse = rmse(val_y_hat, y[val])
                    val_mape = mean(abs.(val_y_hat .- y[val]) ./ y[val])
                    val_cor = Statistics.cor(val_y_hat, y[val])

                    model_index += 1
                    model_colname = model_colnames[model_index]

                    importances[:, model_index] = impurity_importance(rf_machine.fitresult)

                    input_predictions = @chain DataFrame(
                        unique_col => vcat(shuffled_df[train, unique_col], shuffled_df[val, unique_col]),
                        model_colname => map(x -> round(x; digits = 1), vcat(train_y_hat, val_y_hat))
                    ) begin sort(unique_col) end
                    saved_input_predictions[:, model_index] = input_predictions[:, model_colname]

                    input_pertinences = @chain DataFrame(
                        unique_col => vcat(shuffled_df[train, unique_col], shuffled_df[val, unique_col]),
                        model_colname => vcat(repeat([ "train" ], length(train)),repeat([ "val" ], length(val)))
                    ) begin sort(unique_col) end
                    saved_input_pertinences[:, model_index] = input_pertinences[:, model_colname]

                    ## This block of code deals with external validation, when required
                    if (!isnothing(ext_df)) 
                        test_y_hat = GLM.predict(slopecorrection, DataFrame(:yhat => MLJ.predict(rf_machine, ext_df[:, ext_firstinputcol:end]) ))
                        test_rmse = rmse(test_y_hat, ext_df[:, output_col])
                        test_mape = mean(abs.(test_y_hat .- ext_df[:, output_col]) ./ ext_df[:, output_col])
                        test_cor = Statistics.cor(test_y_hat, ext_df[:, output_col])

                        this_ext_predictions = @chain DataFrame(
                            unique_col => ext_df[:, ext_uniquecol],
                            model_colname => test_y_hat
                        ) begin sort(ext_uniquecol) end
                        saved_ext_predictions[:, model_index] = this_ext_predictions[:, model_colname]
                    else
                        test_rmse = NaN; test_mape = NaN; test_cor = NaN;
                    end

                    verbose && println("Merits calculated")

                    hyperpar_idxes[model_index] = hp_idx
                    replica_idxes[model_index] = rep_idx
                    fold_idxes[model_index] = fold_idx
                    rng_idxes[model_index] = rng_idx
                    train_rmses[model_index] = train_rmse
                    val_rmses[model_index] = val_rmse
                    train_mapes[model_index] = train_mape
                    val_mapes[model_index] = val_mape
                    train_cors[model_index] = train_cor
                    val_cors[model_index] = val_cor


                    if (!isnothing(ext_df)) 
                        test_rmses[model_index] = test_rmse
                        test_mapes[model_index] = test_mape
                        test_cors[model_index] = test_cor
                    else
                        test_rmses[model_index] = NaN
                        test_mapes[model_index] = NaN
                        test_cors[model_index] = NaN
                    end

                    verbose && println("Stuff saved, end of model $(model_index)")

                end
            end

            # push!(ys, y)

        end
    end

    report_df = DataFrame(
        :Hyperpar_Idx => hyperpar_idxes,
        :Replica_Idx => replica_idxes,
        :Fold_Idx => fold_idxes,
        :Rng_Idx => rng_idxes,
        :Train_RMSE=> train_rmses,
        :Val_RMSE => val_rmses,
        :Test_RMSE => test_rmses,
        :Train_MAPE => train_mapes,
        :Val_MAPE=> val_mapes,
        :Test_MAPE => test_mapes,
        :Train_Cor=> train_cors,
        :Val_Cor => val_cors,
        :Test_Cor => test_cors
    )

    if isnothing(custom_input_group)
            
        final_importances = @chain importances begin
            DataFrame(:auto)
            rename!( _, names( _ ) .=> model_colnames )
            hcat( DataFrame(:variable => names(prediction_df[:, input_cols])), _ )
        end

    else

        final_importances = @chain importances begin
            DataFrame(:auto)
            rename!( _, names( _ ) .=> model_colnames )
            hcat( DataFrame(:variable => custom_input_group), _ )
        end

    end

    final_saved_predictions = @chain saved_input_predictions begin
        DataFrame(:auto)
        rename!( _, names( _ ) .=> model_colnames )
        hcat( DataFrame(unique_col => sort(prediction_df[:, unique_col])), _ )
    end

    final_saved_pertinences = @chain saved_input_pertinences begin
        DataFrame(:auto)
        rename!( _, names( _ ) .=> model_colnames )
        hcat( DataFrame(unique_col => sort(prediction_df[:, unique_col])), _ )
    end

    if (!isnothing(ext_df))
        final_external_predictions = @chain saved_ext_predictions begin
            DataFrame(:auto)
            rename!( _, names( _ ) .=> model_colnames )
            hcat( DataFrame(unique_col => sort(ext_df[:, ext_uniquecol])), _ )
        end
    else
        final_external_predictions = nothing
    end

    results = ProbeData(
        name,
        original_df,
        n_folds,
        n_replicas,
        n_rngs,
        report_df,
        final_importances,
        final_saved_predictions,
        final_saved_pertinences,
        final_external_predictions
    )

    return results
end
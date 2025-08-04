struct ProbeData
    name::String
    original_df::DataFrame
    n_folds::Int64
    n_replicas::Int64
    n_rngs::Int64
    merits::DataFrame
    importances::DataFrame
    saved_predictions::DataFrame
    saved_pertinences::DataFrame
    external_predictions::Union{Nothing, DataFrame}
end

function partitionvec(x,stride,i,longtail::Bool=true)
    # longtail=true to lengthen the last entry with the leftovers
    # longtail=false to place the leftovers in their own entry
    stride > 0 || error("stride must be positive") # doesn't handle negative strides
    starts = firstindex(x):stride:(lastindex(x)-longtail*stride)+1 # where to start each subvector
    subvecs = [view(x,starts[i]:get(starts,i+1,lastindex(x)+1)-1) for i in eachindex(starts)]
    train, test = reduce(vcat, [ collect(subvecs[map(x -> x != i, eachindex(subvecs))]) ]...), collect(subvecs[i])
    return train, test
end

function probe_regression_elasticnet(
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
    verbose = true,
    tuning_space = (;
        aa = [ -1 ],
        nodesize_range = [ 5 ],
        min_samples_split = [ 2 ],
        sampsize_range = [ 0.7 ],
        mtry_range = [ -1 ],
        ntrees_range = [ 10 ]
    ),
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
    model_colnames = Symbol.("model_" .* map(x -> lpad(string(x), 6, "0"), collect(1:(n_folds*n_replicas*n_rngs))) )
    hyperpar_idxes = Vector{Int64}(undef, n_folds*n_replicas*n_rngs)
    replica_idxes = Vector{Int64}(undef, n_folds*n_replicas*n_rngs)
    fold_idxes = Vector{Int64}(undef, n_folds*n_replicas*n_rngs)
    rng_idxes = Vector{Int64}(undef, n_folds*n_replicas*n_rngs)
    train_rmses = Vector{Float64}(undef, n_folds*n_replicas*n_rngs)
    val_rmses = Vector{Float64}(undef, n_folds*n_replicas*n_rngs)
    test_rmses = Vector{Float64}(undef, n_folds*n_replicas*n_rngs)
    train_mapes = Vector{Float64}(undef, n_folds*n_replicas*n_rngs)
    val_mapes = Vector{Float64}(undef, n_folds*n_replicas*n_rngs)
    test_mapes = Vector{Float64}(undef, n_folds*n_replicas*n_rngs)
    train_cors = Vector{Float64}(undef, n_folds*n_replicas*n_rngs)
    val_cors = Vector{Float64}(undef, n_folds*n_replicas*n_rngs)
    test_cors = Vector{Float64}(undef, n_folds*n_replicas*n_rngs)
    saved_input_predictions = Matrix{Float64}(undef, nrow(prediction_df), n_folds*n_replicas*n_rngs)
    saved_input_pertinences = Matrix{String}(undef, nrow(prediction_df), n_folds*n_replicas*n_rngs)
    !isnothing(ext_df) && (saved_ext_predictions = Matrix{Float64}(undef, nrow(ext_df), n_folds*n_replicas*n_rngs))

    if isnothing(custom_input_group)
        importances = Matrix{Float64}(undef, length(input_cols), n_folds*n_replicas*n_rngs)
    else
        importances = Matrix{Float64}(undef, length(custom_input_group), n_folds*n_replicas*n_rngs)
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

                    preopt_val_cors = Vector{Float64}()
                    # al_grid = [ (x, y) for x in collect(0.01:0.01:0.99) for y in collect(0.01:0.01:0.99) ]
                    # al_grid = [ (x, y) for x in collect(0.01:0.01:0.99) for y in collect(0.01:0.03:9.99) ]
                    al_grid = [ (x, y) for x in 10 .^ collect(-5:0.1:2) for y in collect(0.01:0.03:9.99) ]

                    for (aa, ll) in al_grid 

                        # α = 0.4  # A balanced mix of L1 and L2 regularization
                        # cvpath = glmnetcv(Matrix(Xtrain), ytrain, alpha=α)

                        # # The glmnetcv function performs cross-validation to choose the best λ value
                        # best_lambda = lambdamin(cvpath)  # λ that minimizes the cross-validation error

                        # Fit the final model using the best λ
                        enet_model = glmnet(Matrix(X[train,:]), y[train], alpha=aa, lambda = [ll])

                        yhat_train = vec(GLMNet.predict(enet_model, Matrix(X[train,:])))
                        yhat_test = vec(GLMNet.predict(enet_model, Matrix(X[val,:])))

                        mae(yhat_train, y[train]) |> mean
                        mae(yhat_test, y[val]) |> mean

                        cor(yhat_train, y[train])

                        push!(preopt_val_cors, cor(yhat_test, y[val]))
                    end

                    print(preopt_val_cors)

                    lookfor = findmax(preopt_val_cors[.!(isnan.(preopt_val_cors))])[1]
                    mxidx = findfirst(preopt_val_cors .== lookfor)
                    mxalpha, mxlambda = al_grid[mxidx]

                    # Fit the final model using the best alpha and λ
                    enet_model = glmnet(Matrix(X[train,:]), y[train], alpha=mxalpha, lambda = [mxlambda])

                    verbose && println("ElasticNet model declared") 

                    verbose && println("Machine built and fit")

                    train_y_hat = vec(GLMNet.predict(enet_model, Matrix(X[train,:])))
                    train_rmse = rmse(train_y_hat, y[train])
                    train_mape = mean(abs.(train_y_hat .- y[train]) ./ y[train])
                    train_cor = Statistics.cor(train_y_hat, y[train])

                    val_y_hat = vec(GLMNet.predict(enet_model, Matrix(X[val,:])))
                    val_rmse = rmse(val_y_hat, y[val])
                    val_mape = mean(abs.(val_y_hat .- y[val]) ./ y[val])
                    val_cor = Statistics.cor(val_y_hat, y[val])

                    model_index += 1
                    model_colname = model_colnames[model_index]

                    importances[:, model_index] = enet_model.betas

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

conf_interval(vv, critical_z=1.96) = critical_z * std(vv) / sqrt(length(vv))

report_regression_merits(m) = combine(
    groupby(m.merits, :Hyperpar_Idx),
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
    :Hyperpar_Idx=> (x-> m.name) => :model
)

function hpimportances(m, hp = 1; change_hashnames=false, hashnamestable::Union{Nothing, DataFrame} = nothing)
    importances = m.importances
    subset_idx = findall(m.merits.Hyperpar_Idx .== hp)

    mean_importances = map(x -> sum(x .* subset_idx)/sum(subset_idx), collect(eachrow(Matrix(importances[:, subset_idx .+ 1]))))
    mean_importances_df = sort( DataFrame(:variable => importances[:,1], :weightedImportance => mean_importances), :weightedImportance; rev=true)

    if change_hashnames
        newnames = leftjoin(mean_importances_df, hashnamestable, on = :variable => :hashname).longname
        mean_importances_df.variable = newnames
    end

    return mean_importances_df
end

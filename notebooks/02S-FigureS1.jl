#####
# Supplementary Figure S1
#####

## Reminder: this code is intended to be `include`d into Figure1.jl, not standalone runnable!

# Convert DataFrame's relevant columns to matrix
dist_data_matrix = Matrix(final_inputs[:,9:end])
# Standardize the data
dist_data_standardized = (dist_data_matrix .- mean(dist_data_matrix, dims=1)) ./ std(dist_data_matrix, dims=1)

# Calculate the Distance Matrices
manhattan_dist = pairwise(Distances.Cityblock(), dist_data_standardized; dims = 1)
correlation_dist = pairwise(Distances.CorrDist(), dist_data_standardized; dims = 1)

rclr_data_matrix = Matrix(final_inputs[:,9:end])
euclidean_dist = pairwise(Distances.Euclidean(), rclr_data_matrix; dims = 1)

# Apply MDS for ordination
manhattan_MDS_results = fit(MDS, manhattan_dist; maxoutdim = 10, distances=true)
correlation_MDS_results = fit(MDS, correlation_dist; maxoutdim = 10, distances=true)
euclidean_MDS_results = fit(MDS, euclidean_dist; maxoutdim = 10, distances=true)

sfig = Figure(size = (1400,1000))

## Manhattan Distances MDS
ax1_1 = Axis(
    sfig[1,1],
    title="Manhattan Distances",
    xlabel="MDS Dimension 1", ylabel="MDS Dimension 2")
sc1_1 = scatter!(
    ax1_1, manhattan_MDS_results.U[:, 1], manhattan_MDS_results.U[:, 2],
    color = final_inputs.fr_score,
    colormap = fr_cgrad)

ax2_1 = Axis(
    sfig[2,1],
    title="Manhattan Distances",
    xlabel="MDS Dimension 1", ylabel="MDS Dimension 2")
sc2_1 = scatter!(
    ax2_1, manhattan_MDS_results.U[:, 1], manhattan_MDS_results.U[:, 2],
    color = final_inputs.zBMI,
    colormap = bmi_cgrad)

ax3_1 = Axis(
    sfig[3,1],
    title="Manhattan Distances",
    xlabel="MDS Dimension 1", ylabel="MDS Dimension 2")
sc3_1 = scatter!(
    ax3_1, manhattan_MDS_results.U[:, 1], manhattan_MDS_results.U[:, 2],
    color = final_inputs.fr_score_age_years,
    colormap = age_cgrad)

## Mahalanobis Distances MDS
ax1_2 = Axis(
    sfig[1,2],
    title="Mahalanobis Distances",
    xlabel="MDS Dimension 1", ylabel="MDS Dimension 2")
sc1_2 = scatter!(
    ax1_2, mahalanobis_MDS_results.U[:, 1], mahalanobis_MDS_results.U[:, 2],
    color = final_inputs.fr_score,
    colormap = fr_cgrad)

ax2_2 = Axis(
    sfig[2,2],
    title="Mahalanobis Distances",
    xlabel="MDS Dimension 1", ylabel="MDS Dimension 2")
sc2_2 = scatter!(
    ax2_2, mahalanobis_MDS_results.U[:, 1], mahalanobis_MDS_results.U[:, 2],
    color = final_inputs.zBMI,
    colormap = bmi_cgrad)

ax3_2 = Axis(
    sfig[3,2],
    title="Mahalanobis Distances",
    xlabel="MDS Dimension 1", ylabel="MDS Dimension 2")
sc3_2 = scatter!(
    ax3_2, mahalanobis_MDS_results.U[:, 1], mahalanobis_MDS_results.U[:, 2],
    color = final_inputs.fr_score_age_years,
    colormap = age_cgrad)

## Correlation Distances MDS
ax1_3 = Axis(
    sfig[1,3],
    title="Correlation Distances",
    xlabel="MDS Dimension 1", ylabel="MDS Dimension 2")
sc1_3 = scatter!(
    ax1_3, correlation_MDS_results.U[:, 1], correlation_MDS_results.U[:, 2],
    color = final_inputs.fr_score,
    colormap = fr_cgrad)

ax2_3 = Axis(
    sfig[2,3],
    title="Correlation Distances",
    xlabel="MDS Dimension 1", ylabel="MDS Dimension 2")
sc2_3 = scatter!(
    ax2_3, correlation_MDS_results.U[:, 1], correlation_MDS_results.U[:, 2],
    color = final_inputs.zBMI,
    colormap = bmi_cgrad)

ax3_3 = Axis(
    sfig[3,3],
    title="Correlation Distances",
    xlabel="MDS Dimension 1", ylabel="MDS Dimension 2")
sc3_3 = scatter!(
    ax3_3, correlation_MDS_results.U[:, 1], correlation_MDS_results.U[:, 2],
    color = final_inputs.fr_score_age_years,
    colormap = age_cgrad)

sfig

save("manuscript/figures/FigS1.png", sfig)

# sc = annotations!(
#     ax, string.(final_inputs.fr_score_age_years), Point.(zip(this_result.U[:,1], this_result.U[:,2]));    
#     color = final_inputs.fr_score_age_years,
#     colormap = :inferno)

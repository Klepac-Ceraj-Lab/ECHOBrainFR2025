using Statistics

function robust_geometric_mean_nonzero(
    x::T where T <: Vector{Float64}
)
    # Filter out zeros
    nonzero_vals = filter(!=(0.0), x)
    if isempty(nonzero_vals)
        error("All values are zero; cannot compute geometric mean.")
    end
    # Compute the mean of the logs of nonzero values
    mean_log = mean(log.(nonzero_vals))
    return exp(mean_log)
end

function rclr_vector(
    x::T where T <:AbstractVector{Float64},
    robust_fun = mean;
    pseudocount=1e-6
    )
    # Adjust for zeros
    x_adjusted = x .+ pseudocount
    # Compute log-transformed values
    log_x = log.(x_adjusted)
    # Compute robust geometric mean explicitly
    robust_geom_mean = exp(robust_fun(log_x))
    # Perform the rCLR transformation
    return log_x .- log(robust_geom_mean)
end

ordered_brain_names = [

    "total_intracranial",
    
    "left_cerebral_white_matter",
    "right_cerebral_white_matter",
    "left_cerebral_cortex",
    "right_cerebral_cortex",

    "left_cerebellum_white_matter",
    "right_cerebellum_white_matter",
    "left_cerebellum_cortex",
    "right_cerebellum_cortex",
    "csf", 
    
    "3rd_ventricle",
    "4th_ventricle",
    "brain-stem",

    "left_thalamus",
    "right_thalamus",
    "left_caudate",
    "right_caudate",
    "left_putamen",
    "right_putamen",
    "left_pallidum",
    "right_pallidum",

    "left_hippocampus",
    "right_hippocampus",
    "left_amygdala",
    "right_amygdala",
    "left_accumbens_area",
    "right_accumbens_area",
    "left_ventral_DC",
    "right_ventral_DC",
    "left_lateral_ventricle",
    "right_lateral_ventricle",
    "left_inferior_lateral_ventricle",
    "right_inferior_lateral_ventricle",
    
    "ctx-lh-bankssts",
    "ctx-rh-bankssts",
    "ctx-lh-caudalanteriorcingulate",
    "ctx-rh-caudalanteriorcingulate",
    "ctx-lh-caudalmiddlefrontal",
    "ctx-rh-caudalmiddlefrontal",
    "ctx-lh-cuneus",
    "ctx-rh-cuneus",
    "ctx-lh-entorhinal",
    "ctx-rh-entorhinal",
    "ctx-lh-fusiform",
    "ctx-rh-fusiform",
    "ctx-lh-inferiorparietal",
    "ctx-rh-inferiorparietal",
    "ctx-lh-inferiortemporal",
    "ctx-rh-inferiortemporal",
    "ctx-lh-isthmuscingulate",
    "ctx-rh-isthmuscingulate",
    "ctx-lh-lateraloccipital",
    "ctx-rh-lateraloccipital",
    "ctx-lh-lateralorbitofrontal",
    "ctx-rh-lateralorbitofrontal",
    "ctx-lh-lingual",
    "ctx-rh-lingual",
    "ctx-lh-medialorbitofrontal",
    "ctx-rh-medialorbitofrontal",
    "ctx-lh-middletemporal",
    "ctx-rh-middletemporal",
    "ctx-lh-parahippocampal",
    "ctx-rh-parahippocampal",
    "ctx-lh-paracentral",
    "ctx-rh-paracentral",
    "ctx-lh-parsopercularis",
    "ctx-rh-parsopercularis",
    "ctx-lh-parsorbitalis",
    "ctx-rh-parsorbitalis",
    "ctx-lh-parstriangularis",
    "ctx-rh-parstriangularis",
    "ctx-lh-pericalcarine",
    "ctx-rh-pericalcarine",
    "ctx-lh-postcentral",
    "ctx-rh-postcentral",
    "ctx-lh-posteriorcingulate",
    "ctx-rh-posteriorcingulate",
    "ctx-lh-precentral",
    "ctx-rh-precentral",
    "ctx-lh-precuneus",
    "ctx-rh-precuneus",
    "ctx-lh-rostralanteriorcingulate",
    "ctx-rh-rostralanteriorcingulate",
    "ctx-lh-rostralmiddlefrontal",
    "ctx-rh-rostralmiddlefrontal",
    "ctx-lh-superiorfrontal",
    "ctx-rh-superiorfrontal",
    "ctx-lh-superiorparietal",
    "ctx-rh-superiorparietal",
    "ctx-lh-superiortemporal",
    "ctx-rh-superiortemporal",
    "ctx-lh-supramarginal",
    "ctx-rh-supramarginal",
    "ctx-lh-frontalpole",
    "ctx-rh-frontalpole",
    "ctx-lh-temporalpole",
    "ctx-rh-temporalpole",
    "ctx-lh-transversetemporal",
    "ctx-rh-transversetemporal",
    "ctx-lh-insula",
    "ctx-rh-insula"
]

ordered_brain_renamepairs = [

    # "total_intracranial",
    
    # "left_cerebral_white_matter",
    # "right_cerebral_white_matter",
    # "left_cerebral_cortex",
    # "right_cerebral_cortex",

    # "left_cerebellum_white_matter",
    # "right_cerebellum_white_matter",
    # "left_cerebellum_cortex",
    # "right_cerebellum_cortex",
    # "csf", 
    
    # "3rd_ventricle",
    # "4th_ventricle",
    # "brain-stem",

    # "left_thalamus",
    # "right_thalamus",
    # "left_caudate",
    # "right_caudate",
    # "left_putamen",
    # "right_putamen",
    # "left_pallidum",
    # "right_pallidum",

    # "left_hippocampus",
    # "right_hippocampus",
    # "left_amygdala",
    # "right_amygdala",
    # "left_accumbens_area",
    # "right_accumbens_area",
    # "left_ventral_DC",
    # "right_ventral_DC",
    # "left_lateral_ventricle",
    # "right_lateral_ventricle",
    # "left_inferior_lateral_ventricle",
    # "right_inferior_lateral_ventricle",
    
    "ctx-lh-bankssts" => "left_bankssts",
    "ctx-rh-bankssts" => "right_bankssts",
    "ctx-lh-caudalanteriorcingulate" => "left_caudal_anterior_cingulate",
    "ctx-rh-caudalanteriorcingulate" => "right_caudal_anterior_cingulate",
    "ctx-lh-caudalmiddlefrontal" => "left_caudal_middle_frontal",
    "ctx-rh-caudalmiddlefrontal" => "right_caudal_middle_frontal",
    "ctx-lh-cuneus" => "left_cuneus",
    "ctx-rh-cuneus" => "right_cuneus",
    "ctx-lh-entorhinal" => "left_entorhinal",
    "ctx-rh-entorhinal" => "right_entorhinal",
    "ctx-lh-fusiform" => "left_fusiform",
    "ctx-rh-fusiform" => "right_fusiform",
    "ctx-lh-inferiorparietal" => "left_inferior_parietal",
    "ctx-rh-inferiorparietal" => "right_inferior_parietal",
    "ctx-lh-inferiortemporal" => "left_inferior_temporal",
    "ctx-rh-inferiortemporal" => "right_inferior_temporal",
    "ctx-lh-isthmuscingulate" => "left_isthmus_cingulate",
    "ctx-rh-isthmuscingulate" => "right_isthmus_cingulate",
    "ctx-lh-lateraloccipital" => "left_lateral_occipital",
    "ctx-rh-lateraloccipital" => "right_lateral_occipital",
    "ctx-lh-lateralorbitofrontal" => "left_lateral_orbitofrontal",
    "ctx-rh-lateralorbitofrontal" => "right_lateral_orbitofrontal",
    "ctx-lh-lingual" => "left_lingual",
    "ctx-rh-lingual" => "right_lingual",
    "ctx-lh-medialorbitofrontal" => "left_medial_orbitofrontal",
    "ctx-rh-medialorbitofrontal" => "right_medial_orbitofrontal",
    "ctx-lh-middletemporal" => "left_middle_temporal",
    "ctx-rh-middletemporal" => "right_middle_temporal",
    "ctx-lh-parahippocampal" => "left_parahippocampal",
    "ctx-rh-parahippocampal" => "right_parahippocampal",
    "ctx-lh-paracentral" => "left_paracentral",
    "ctx-rh-paracentral" => "right_paracentral",
    "ctx-lh-parsopercularis" => "left_pars_opercularis",
    "ctx-rh-parsopercularis" => "right_pars_opercularis",
    "ctx-lh-parsorbitalis" => "left_pars_orbitalis",
    "ctx-rh-parsorbitalis" => "right_pars_orbitalis",
    "ctx-lh-parstriangularis" => "left_pars_triangularis",
    "ctx-rh-parstriangularis" => "right_pars_triangularis",
    "ctx-lh-pericalcarine" => "left_pericalcarine",
    "ctx-rh-pericalcarine" => "right_pericalcarine",
    "ctx-lh-postcentral" => "left_postcentral",
    "ctx-rh-postcentral" => "right_postcentral",
    "ctx-lh-posteriorcingulate" => "left_posterior_cingulate",
    "ctx-rh-posteriorcingulate" => "right_posterior_cingulate",
    "ctx-lh-precentral" => "left_precentral",
    "ctx-rh-precentral" => "right_precentral",
    "ctx-lh-precuneus" => "left_precuneus",
    "ctx-rh-precuneus" => "right_precuneus",
    "ctx-lh-rostralanteriorcingulate" => "left_rostral_anterior_cingulate",
    "ctx-rh-rostralanteriorcingulate" => "right_rostral_anterior_cingulate",
    "ctx-lh-rostralmiddlefrontal" => "left_rostral_middle_frontal",
    "ctx-rh-rostralmiddlefrontal" => "right_rostral_middle_frontal",
    "ctx-lh-superiorfrontal" => "left_superior_frontal",
    "ctx-rh-superiorfrontal" => "right_superior_frontal",
    "ctx-lh-superiorparietal" => "left_superior_parietal",
    "ctx-rh-superiorparietal" => "right_superior_parietal",
    "ctx-lh-superiortemporal" => "left_superior_temporal",
    "ctx-rh-superiortemporal" => "right_superior_temporal",
    "ctx-lh-supramarginal" => "left_supramarginal",
    "ctx-rh-supramarginal" => "right_supramarginal",
    "ctx-lh-frontalpole" => "left_frontal_pole",
    "ctx-rh-frontalpole" => "right_frontal_pole",
    "ctx-lh-temporalpole" => "left_temporal_pole",
    "ctx-rh-temporalpole" => "right_temporal_pole",
    "ctx-lh-transversetemporal" => "left_transverse_temporal",
    "ctx-rh-transversetemporal" => "right_transverse_temporal",
    "ctx-lh-insula" => "left_insula",
    "ctx-rh-insula" => "right_insula"
]

ordered_brain_newnames = [
    ## This should be the new selection. Include cerebellum cortex, exclude ventricles.
    # "total_intracranial",
    # "left_cerebral_white_matter", "right_cerebral_white_matter",
    # "left_cerebral_cortex", "right_cerebral_cortex",
    # "left_cerebellum_white_matter", "right_cerebellum_white_matter",
    # "left_cerebellum_cortex", "right_cerebellum_cortex",
    # "csf",
    # "3rd_ventricle", "4th_ventricle",
    # "brain-stem",
    "left_thalamus", "right_thalamus",
    "left_caudate", "right_caudate",
    "left_putamen", "right_putamen",
    "left_pallidum", "right_pallidum",
    "left_hippocampus", "right_hippocampus",
    "left_amygdala", "right_amygdala",
    "left_accumbens_area", "right_accumbens_area",
    "left_ventral_DC", "right_ventral_DC",
    # "left_lateral_ventricle", "right_lateral_ventricle",
    # "left_inferior_lateral_ventricle", "right_inferior_lateral_ventricle",
    "left_bankssts", "right_bankssts",
    "left_caudal_anterior_cingulate", "right_caudal_anterior_cingulate",
    "left_caudal_middle_frontal", "right_caudal_middle_frontal",
    "left_cuneus", "right_cuneus",
    "left_entorhinal", "right_entorhinal",
    "left_fusiform", "right_fusiform",
    "left_inferior_parietal", "right_inferior_parietal",
    "left_inferior_temporal", "right_inferior_temporal",
    "left_isthmus_cingulate", "right_isthmus_cingulate",
    "left_lateral_occipital", "right_lateral_occipital",
    "left_lateral_orbitofrontal", "right_lateral_orbitofrontal",
    "left_lingual", "right_lingual",
    "left_medial_orbitofrontal", "right_medial_orbitofrontal",
    "left_middle_temporal", "right_middle_temporal",
    "left_parahippocampal", "right_parahippocampal",
    "left_paracentral", "right_paracentral",
    "left_pars_opercularis", "right_pars_opercularis",
    "left_pars_orbitalis", "right_pars_orbitalis",
    "left_pars_triangularis", "right_pars_triangularis",
    "left_pericalcarine", "right_pericalcarine",
    "left_postcentral", "right_postcentral",
    "left_posterior_cingulate", "right_posterior_cingulate",
    "left_precentral", "right_precentral",
    "left_precuneus", "right_precuneus",
    "left_rostral_anterior_cingulate", "right_rostral_anterior_cingulate",
    "left_rostral_middle_frontal", "right_rostral_middle_frontal",
    "left_superior_frontal", "right_superior_frontal",
    "left_superior_parietal", "right_superior_parietal",
    "left_superior_temporal", "right_superior_temporal",
    "left_supramarginal", "right_supramarginal",
    "left_frontal_pole", "right_frontal_pole",
    "left_temporal_pole", "right_temporal_pole",
    "left_transverse_temporal", "right_transverse_temporal",
    "left_insula", "right_insula"
]

function autonormalize(vv; exclude_zeros = false) 
    if exclude_zeros
        this_mean = mean(vv[vv .> 0.0])
        this_stdev = Statistics.std(vv[vv .> 0.0])
    else
        this_mean = mean(vv)
        this_stdev = Statistics.std(vv)
    end

    (this_stdev == 0) && return(NaN)

    vv = ( vv .- this_mean ) ./ this_stdev

    return vv
end

function rangenormalize(vv) 
    this_max = maximum(vv)
    this_min = minimum(vv)

    (this_min == this_max) && return(NaN)

    vv = ( vv .- this_min ) ./ (this_max - this_min)
    return vv
end

#####
# This script loads and joins the Metadata from Appetite Lab (S.Carnell) to the brain volumes from S.Deoni
#####

using CSV, DataFrames, Chain
using CategoricalArrays

# metdata_xf_path = joinpath("data", "2025-03-14-BrainFRCompleteMetadata.xlsx")
# metadata_xf = XLSX.readxlsx(metdata_xf_path)
# metadata_xf_sheetname = XLSX.sheetnames(metadata_xf)[1]
# metadata_xf_sheet = metadata_xf[metadata_xf_sheetname]
# metadata_df = DataFrame(XLSX.readtable(metdata_xf_path, sheetname))
## Sigh... type comes back as Any if we do this... converting to CSV and moving on...

function load_brain_data()
    ## Load and process FR metadata
    metadata_df_raw = CSV.read(joinpath("data", "2025-03-14-BrainFRCompleteMetadata.csv"), DataFrame; stringtype = String)
    metadata_df = @chain metadata_df_raw begin
        dropmissing(:ageyears)
        subset(:ageyears => (x -> x .!= "#NULL!"))
        dropmissing(:sex)
        subset(:sex => (x -> x .!= "#NULL!"))
        dropmissing(:WHO_all_zbmi)
        subset(:WHO_all_zbmi => (x -> x .!= "#NULL!"))
        dropmissing(:CEBQfoodResponsivenenss)
        subset(:CEBQfoodResponsivenenss => (x -> x .!= "#NULL!"))
        dropmissing(:Mom_education_2cat)
        subset(:Mom_education_2cat => (x -> x .!= "#NULL!"))
        rename!(
            :ageyears => :fr_score_age_years,
            :WHO_all_zbmi => :zBMI,
            :CEBQfoodResponsivenenss => :fr_score,
            :Mom_education_2cat => :mom_edu_cat
        )
        DataFrames.transform!(:fr_score => (x -> parse.(Float64, x)) => :fr_score)
        DataFrames.transform!(:sex => (x -> CategoricalArray(x, ordered = false)) => :sex)
        DataFrames.transform!(:mom_edu_cat => (x -> CategoricalArray(x, ordered = true)) => :mom_edu_cat)
        unique([:studyID, :timepoint])
    end

    ## Load and process brain data
    brain_df_raw = CSV.read(joinpath("data", "2025-01-22-ECHONewBrainVolumes.csv"), DataFrame; stringtype = String)
    brain_df = @chain brain_df_raw begin
        select(Not([
            "study_childSex",
            "study_childAge",
            "study_maternalSES",
            "study_childBirthWeight",
            "study_childCurrentWeight_kgs",
            "study_childCurrentHeight_cms",
            "study_MSEL_ELC",
            "study_MSEL_VDQ",
            "study_MSEL_NVDQ",
            "BasicID",
            "subject",
        ]))
        unique([:StudyID, :Timepoint])
    end

    ## Joining both tables and getting final set of samples for study_childAge

    joined_data = innerjoin(
        metadata_df,
        brain_df,
        on = [:studyID => :StudyID, :timepoint => :Timepoint]
    )

    joined_input_data = @chain joined_data begin
        sort([:studyID, :timepoint]; rev=true) ## This makes the code pick up the latest timepoint
        # sort([:studyID, :timepoint]) ## This makes the code pick up the earliest timepoint
        unique(:studyID)
        DataFrames.transform([:studyID, :timepoint] => ( (x,y) -> ["$(el[1])-$(el[2])" for el in zip(x,y)]) => :UID; renamecols = false)
    end

    ##########################################
    # BLOCK TO REMOVE OUTLIERS BASED ON AGES #
    ##########################################

    exclude_ids = [ 925, 768, 259]
    excluded_input_data = subset(joined_input_data, :studyID => x -> x .∈ Ref(exclude_ids))

    filtered_input_data = subset(joined_input_data, :studyID => x -> x .∉ Ref(exclude_ids))

    input_brain =  @chain filtered_input_data begin
        select( "UID", ordered_brain_names)
        rename(ordered_brain_renamepairs)
    end
    
    input_mdata = @chain filtered_input_data begin
        select(:UID, :)
        select(Not(ordered_brain_names))
    end

    return(input_mdata, input_brain)

end
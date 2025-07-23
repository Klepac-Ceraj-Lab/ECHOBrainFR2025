## Preprocessing brain
input_brain = @chain input_brain_raw begin
    DataFrames.transform( _ , ordered_brain_newnames .=> (y -> y ./ _.total_intracranial) .=> ordered_brain_newnames; renamecols=false)
    # DataFrames.transform( _ , ordered_brain_newnames .=> (y -> y ./ _.total_intracranial) .=> ordered_brain_newnames; renamecols=false)
    # DataFrames.transform( ordered_brain_newnames .=> (y -> autonormalize(y) ) .=> ordered_brain_newnames; renamecols=false)
    select("UID", ordered_brain_newnames)
end

input_brain_rclr = deepcopy(input_brain)

for i in 1:nrow(input_brain)
    input_brain_rclr[i,2:end] = rclr_vector(collect(input_brain_rclr[i,2:end]))
end

final_inputs = @chain input_mdata begin 
    innerjoin(input_brain, on = :UID)
    select(["UID", "studyID", "timepoint", "fr_score", "zBMI", "mom_edu_cat", "sex", "fr_score_age_years" ], ordered_brain_newnames)
    transform!(:sex => ( x -> levelcode.(x) ) => :sex)
    transform!(:mom_edu_cat => ( x -> levelcode.(x) ) => :mom_edu_cat)
end

final_inputs_rclr = @chain input_mdata begin 
    innerjoin(input_brain_rclr, on = :UID)
    select(["UID", "studyID", "timepoint", "fr_score", "zBMI", "mom_edu_cat", "sex", "fr_score_age_years" ], ordered_brain_newnames)
    transform!(:sex => ( x -> levelcode.(x) ) => :sex)
    transform!(:mom_edu_cat => ( x -> levelcode.(x) ) => :mom_edu_cat)
end

CSV.write(joinpath("results", "input_data.csv"), final_inputs)
CSV.write(joinpath("results", "input_data_rclr.csv"), final_inputs_rclr)
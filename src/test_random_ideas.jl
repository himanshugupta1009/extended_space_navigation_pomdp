function lala()
    folder_location = "./scenario_3/"
    #Delete output txt files
    #1D planner
    foreach(rm, filter(endswith(".txt"), readdir(folder_location*"1D",join=true)))
    #2D planner
    foreach(rm, filter(endswith(".txt"), readdir(folder_location*"2D",join=true)))

    #Delete jld2 files
    #1D planner
    foreach(rm, filter(endswith(".jld2"), readdir(folder_location*"1D/risky_scenarios",join=true)))
    #2D planner
    foreach(rm, filter(endswith(".jld2"), readdir(folder_location*"2D/risky_scenarios",join=true)))
end

lala()

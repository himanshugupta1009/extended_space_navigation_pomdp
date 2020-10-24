function see_human_motion(all_worlds, human_num)
    for i in 1:length(all_worlds)
        println(all_worlds[i].humans[human_num])
    end
end



module A
    include("environment.jl")
    export hg, display_env

    function hg()
        display_env(env)
        @show("HI")
    end
end

##display_env(env)

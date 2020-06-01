using PackageCompiler
create_sysimage([:iLQGames, :Plots], sysimage_path="$(@__DIR__)/ilqgames_dev.sysimg.so",
                precompile_execution_file="$(@__DIR__)/precompile.jl")

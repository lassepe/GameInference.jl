const default_font = Plots.font("Times", 10)
const default_font_small = Plots.font("Times", 8)

gr(size=(600,300),
   titlefont=default_font,
   guidefont=default_font,
   legendfont=default_font_small,
   tickfont=default_font_small,
   markerstrokecolor=:white,
   dpi=75)

id2figchar(i; start_char='a') = "($(start_char+i-1))"

function plot_median_with_ribbon(args...; kwargs...)
    return plot_median_with_ribbon!(plot(), args...; kwargs...)
end

function plot_median_with_ribbon!(plt, df, x_sym, y_sym; kwargs...)
    # compute the statistics on the  padded data
    sem(x) = std(x) / sqrt(length(x))
    qe(x, p) = abs(median(x) - quantile(x, p))

    df_stats = @linq df |>
        by(x_sym,
           y_median=median(cols(y_sym)),
           y_low=qe(cols(y_sym), 0.25),
           y_high=qe(cols(y_sym), 0.75))

    return @df df_stats plot!(plt, cols(x_sym), :y_median; ribbon=(:y_low, :y_high),
                              kwargs...)
end

# construct a relaxed grid of plots
function plot_rowise(plts; n_cols=3, size_per_subplot=(200, 200), kwargs...)
    kwargs_with_defaults = merge((margin=3Plots.mm,),
                                 kwargs)

    plt_rows = []
    for i in 1:n_cols:length(plts)
        subplts = plts[i:min(i+n_cols-1, length(plts))]
        p = plot(subplts...; layout=(1, length(subplts)))
        push!(plt_rows, p)
    end

    return plot(plt_rows...;
                layout=(length(plt_rows), 1),
                size=(size_per_subplot.* (n_cols, length(plt_rows))),
                kwargs_with_defaults...)
end

function saveplts(plts_with_anno::AbstractArray{<:Tuple{<:Plots.Plot,
                                                      <:AbstractString}},
                  datadir::AbstractString, file_prefix::AbstractString)

    known_files = []
    for (p, anno) in plts_with_anno
        filename = joinpath(datadir, "$file_prefix-$anno.pdf")
        @assert !(filename in known_files)
        savefig(p, filename)
    end
end

function latexified_cost_summary(df::DataFrame)
    sem(x) = std(x) / sqrt(length(x))
    mean_sem_string(x; digits=1) = string(round(mean(x); digits=digits),
                                               " \\pm ",
                                               round(sem(x); digits=digits))
    # create raw summery
    df_summary = @linq df |> by([:ego, :player_id],
                                cost=mean_sem_string(:cost))

    # make a new dataframes, where there is a column for each player and a row for
    # each approach
    df_compact = unstack(df_summary, :ego, :player_id, :cost; renamecols=x->"{mean cost P$x}")

    return latexify(df_compact[:, 2:end]; env=:tabular, latex=false, side=df_compact[:, 1])
end


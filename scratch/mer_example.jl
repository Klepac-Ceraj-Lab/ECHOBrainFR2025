using CairoMakie

fig = Figure(size = (1000,1000))

axA = Axis(
    fig[1,1],
    xlabel = "Feed Type",
    ylabel = "CT Values Bifido"
)

axB = Axis(
    fig[1,2],
    xlabel = "Feed Type",
    ylabel = "CT Values Bifido"
)

axC = Axis(
    fig[2,1],
    xlabel = "Feed Type",
    ylabel = "CT Values Bifido"
)

axD = Axis(
    fig[2,2],
    xlabel = "Feed Type",
    ylabel = "CT Values Bifido"
)
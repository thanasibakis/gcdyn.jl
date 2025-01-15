using Documenter, DocumenterInterLinks, gcdyn
import StatsAPI, Plots

external_links = InterLinks(
	"ColorSchemes" => "https://juliagraphics.github.io/ColorSchemes.jl/stable/objects.inv"
)

makedocs(
	sitename="gcdyn.jl",
	modules = [gcdyn],
	plugins = [external_links]
)

deploydocs(
    repo = "github.com/thanasibakis/gcdyn.jl.git",
)